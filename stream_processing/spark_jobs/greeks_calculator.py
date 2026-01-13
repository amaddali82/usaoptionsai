"""
Options Greeks Calculator for Streaming Data
Real-time calculation of option Greeks using Spark
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import numpy as np
from scipy.stats import norm
import logging


# UDFs for Greeks calculation
def calculate_d1(S, K, T, r, sigma, q=0):
    """Calculate d1 parameter for Black-Scholes"""
    return (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

def calculate_d2(S, K, T, r, sigma, q=0):
    """Calculate d2 parameter"""
    return calculate_d1(S, K, T, r, sigma, q) - sigma * np.sqrt(T)

def bs_delta(S, K, T, r, sigma, option_type, q=0):
    """Calculate option delta"""
    if T <= 0:
        return 1.0 if S > K and option_type == 'call' else 0.0
    d1 = calculate_d1(S, K, T, r, sigma, q)
    if option_type == 'call':
        return float(np.exp(-q * T) * norm.cdf(d1))
    else:
        return float(-np.exp(-q * T) * norm.cdf(-d1))

def bs_gamma(S, K, T, r, sigma, q=0):
    """Calculate option gamma"""
    if T <= 0:
        return 0.0
    d1 = calculate_d1(S, K, T, r, sigma, q)
    return float(np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T)))

def bs_vega(S, K, T, r, sigma, q=0):
    """Calculate option vega"""
    if T <= 0:
        return 0.0
    d1 = calculate_d1(S, K, T, r, sigma, q)
    return float(S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100)

def bs_theta(S, K, T, r, sigma, option_type, q=0):
    """Calculate option theta"""
    if T <= 0:
        return 0.0
    d1 = calculate_d1(S, K, T, r, sigma, q)
    d2 = calculate_d2(S, K, T, r, sigma, q)
    
    if option_type == 'call':
        theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)
                 + q * S * np.exp(-q * T) * norm.cdf(d1))
    else:
        theta = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)
                 - q * S * np.exp(-q * T) * norm.cdf(-d1))
    
    return float(theta / 365)  # Per day


class GreeksCalculatorJob:
    """Spark streaming job for real-time Greeks calculation"""
    
    def __init__(self, kafka_brokers: str = "localhost:9092",
                 checkpoint_dir: str = "/tmp/spark-greeks-checkpoints"):
        """Initialize Greeks calculator job"""
        self.kafka_brokers = kafka_brokers
        self.checkpoint_dir = checkpoint_dir
        self.logger = logging.getLogger(__name__)
        
        # Create Spark session
        self.spark = SparkSession.builder \
            .appName("OptionsAI-GreeksCalculator") \
            .config("spark.sql.streaming.checkpointLocation", checkpoint_dir) \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        
        # Register UDFs
        self.register_udfs()
        
        self.logger.info("Greeks calculator initialized")
    
    def register_udfs(self):
        """Register user-defined functions for Greeks"""
        
        # Delta UDF
        delta_udf = udf(bs_delta, DoubleType())
        self.spark.udf.register("calculate_delta", delta_udf)
        
        # Gamma UDF
        gamma_udf = udf(bs_gamma, DoubleType())
        self.spark.udf.register("calculate_gamma", gamma_udf)
        
        # Vega UDF
        vega_udf = udf(bs_vega, DoubleType())
        self.spark.udf.register("calculate_vega", vega_udf)
        
        # Theta UDF
        theta_udf = udf(bs_theta, DoubleType())
        self.spark.udf.register("calculate_theta", theta_udf)
        
        self.logger.info("UDFs registered")
    
    def create_options_stream(self):
        """Create streaming DataFrame from options Kafka topic"""
        
        # Define schema
        options_schema = StructType([
            StructField("symbol", StringType(), False),
            StructField("options_ticker", StringType(), False),
            StructField("strike", DoubleType(), False),
            StructField("expiration", StringType(), False),
            StructField("option_type", StringType(), False),
            StructField("underlying_price", DoubleType(), False),
            StructField("implied_volatility", DoubleType(), True),
            StructField("timestamp", StringType(), False)
        ])
        
        # Read from Kafka
        df = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_brokers) \
            .option("subscribe", "raw.options.quotes") \
            .option("startingOffsets", "latest") \
            .load()
        
        # Parse JSON
        parsed_df = df.select(
            from_json(col("value").cast("string"), options_schema).alias("data")
        ).select("data.*")
        
        return parsed_df
    
    def calculate_all_greeks(self, df, risk_free_rate: float = 0.05, dividend_yield: float = 0.02):
        """
        Calculate all Greeks for options
        
        Args:
            df: Streaming DataFrame with options data
            risk_free_rate: Annual risk-free rate
            dividend_yield: Annual dividend yield
        """
        
        # Calculate time to expiration
        df_with_tte = df.withColumn(
            "time_to_expiration",
            (unix_timestamp(col("expiration")) - unix_timestamp(col("timestamp"))) / (365.25 * 24 * 3600)
        )
        
        # Calculate Greeks using UDFs
        df_with_greeks = df_with_tte \
            .withColumn("delta", 
                expr(f"calculate_delta(underlying_price, strike, time_to_expiration, {risk_free_rate}, implied_volatility, option_type, {dividend_yield})")) \
            .withColumn("gamma",
                expr(f"calculate_gamma(underlying_price, strike, time_to_expiration, {risk_free_rate}, implied_volatility, {dividend_yield})")) \
            .withColumn("vega",
                expr(f"calculate_vega(underlying_price, strike, time_to_expiration, {risk_free_rate}, implied_volatility, {dividend_yield})")) \
            .withColumn("theta",
                expr(f"calculate_theta(underlying_price, strike, time_to_expiration, {risk_free_rate}, implied_volatility, option_type, {dividend_yield})"))
        
        # Add moneyness
        df_final = df_with_greeks \
            .withColumn("moneyness", col("underlying_price") / col("strike")) \
            .withColumn("intrinsic_value", 
                when(col("option_type") == "call", 
                     greatest(col("underlying_price") - col("strike"), lit(0)))
                .otherwise(greatest(col("strike") - col("underlying_price"), lit(0))))
        
        return df_final
    
    def process_and_write_greeks(self):
        """Main processing pipeline for Greeks"""
        
        self.logger.info("Starting Greeks calculation pipeline")
        
        # Create input stream
        options_stream = self.create_options_stream()
        
        # Calculate Greeks
        greeks = self.calculate_all_greeks(options_stream)
        
        # Add processing metadata
        greeks_with_meta = greeks \
            .withColumn("calculation_time", current_timestamp())
        
        # Write to console (debugging)
        query_console = greeks_with_meta \
            .writeStream \
            .outputMode("append") \
            .format("console") \
            .option("truncate", False) \
            .start()
        
        # Write to Kafka
        query_kafka = greeks_with_meta \
            .selectExpr("options_ticker as key", "to_json(struct(*)) as value") \
            .writeStream \
            .outputMode("append") \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_brokers) \
            .option("topic", "processed.greeks") \
            .option("checkpointLocation", f"{self.checkpoint_dir}/kafka") \
            .start()
        
        self.logger.info("Greeks calculation pipeline started")
        
        # Wait for termination
        query_console.awaitTermination()
        query_kafka.awaitTermination()
    
    def stop(self):
        """Stop Spark session"""
        self.logger.info("Stopping Spark session")
        self.spark.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    job = GreeksCalculatorJob(kafka_brokers="localhost:9092")
    
    try:
        job.process_and_write_greeks()
    except KeyboardInterrupt:
        job.stop()
