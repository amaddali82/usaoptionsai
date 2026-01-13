"""
Apache Spark Structured Streaming Job for Feature Engineering
Consumes data from Kafka, calculates technical indicators, and stores in databases
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import logging


class FeatureExtractionJob:
    """Spark streaming job for real-time feature extraction"""
    
    def __init__(self, kafka_brokers: str = "localhost:9092",
                 checkpoint_dir: str = "/tmp/spark-checkpoints"):
        """
        Initialize Spark streaming job
        
        Args:
            kafka_brokers: Comma-separated Kafka broker addresses
            checkpoint_dir: Directory for Spark checkpoints
        """
        self.kafka_brokers = kafka_brokers
        self.checkpoint_dir = checkpoint_dir
        self.logger = logging.getLogger(__name__)
        
        # Create Spark session
        self.spark = SparkSession.builder \
            .appName("OptionsAI-FeatureExtraction") \
            .config("spark.sql.streaming.checkpointLocation", checkpoint_dir) \
            .config("spark.sql.shuffle.partitions", "6") \
            .getOrCreate()
        
        self.spark.sparkContext.setLogLevel("WARN")
        self.logger.info("Spark session initialized")
    
    def create_stock_price_stream(self):
        """Create streaming DataFrame from stock price Kafka topic"""
        
        # Define schema for stock price data
        price_schema = StructType([
            StructField("symbol", StringType(), False),
            StructField("timestamp", StringType(), False),
            StructField("data", StructType([
                StructField("open", DoubleType(), False),
                StructField("high", DoubleType(), False),
                StructField("low", DoubleType(), False),
                StructField("close", DoubleType(), False),
                StructField("volume", LongType(), False)
            ]), False)
        ])
        
        # Read from Kafka
        df = self.spark \
            .readStream \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_brokers) \
            .option("subscribe", "raw.stock.prices") \
            .option("startingOffsets", "latest") \
            .load()
        
        # Parse JSON
        parsed_df = df.select(
            from_json(col("value").cast("string"), price_schema).alias("data")
        ).select("data.*")
        
        # Flatten nested structure
        flattened_df = parsed_df.select(
            col("symbol"),
            to_timestamp(col("timestamp")).alias("timestamp"),
            col("data.open").alias("open"),
            col("data.high").alias("high"),
            col("data.low").alias("low"),
            col("data.close").alias("close"),
            col("data.volume").alias("volume")
        )
        
        return flattened_df
    
    def calculate_technical_indicators(self, df):
        """
        Calculate technical indicators from price data
        
        Args:
            df: Streaming DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        from pyspark.sql.window import Window
        
        # Define window for calculations (by symbol, ordered by time)
        window_spec = Window.partitionBy("symbol").orderBy("timestamp")
        
        # Add row number for moving averages
        df_with_row = df.withColumn("row_num", row_number().over(window_spec))
        
        # Simple Moving Averages (SMA)
        sma_5_window = window_spec.rowsBetween(-4, 0)
        sma_10_window = window_spec.rowsBetween(-9, 0)
        sma_20_window = window_spec.rowsBetween(-19, 0)
        sma_50_window = window_spec.rowsBetween(-49, 0)
        
        df_with_sma = df_with_row \
            .withColumn("sma_5", avg("close").over(sma_5_window)) \
            .withColumn("sma_10", avg("close").over(sma_10_window)) \
            .withColumn("sma_20", avg("close").over(sma_20_window)) \
            .withColumn("sma_50", avg("close").over(sma_50_window))
        
        # Price momentum
        df_with_momentum = df_with_sma \
            .withColumn("return_1d", (col("close") - lag("close", 1).over(window_spec)) / lag("close", 1).over(window_spec)) \
            .withColumn("return_5d", (col("close") - lag("close", 5).over(window_spec)) / lag("close", 5).over(window_spec)) \
            .withColumn("return_20d", (col("close") - lag("close", 20).over(window_spec)) / lag("close", 20).over(window_spec))
        
        # Volatility (rolling standard deviation)
        vol_window = window_spec.rowsBetween(-19, 0)
        df_with_volatility = df_with_momentum \
            .withColumn("volatility_20d", stddev("return_1d").over(vol_window))
        
        # Volume indicators
        vol_avg_window = window_spec.rowsBetween(-19, 0)
        df_with_vol_indicators = df_with_volatility \
            .withColumn("volume_sma_20", avg("volume").over(vol_avg_window)) \
            .withColumn("volume_ratio", col("volume") / col("volume_sma_20"))
        
        # Price range
        df_final = df_with_vol_indicators \
            .withColumn("price_range", col("high") - col("low")) \
            .withColumn("price_range_pct", (col("high") - col("low")) / col("close"))
        
        return df_final
    
    def calculate_rsi(self, df, period: int = 14):
        """
        Calculate Relative Strength Index (RSI)
        
        Args:
            df: DataFrame with price data
            period: RSI period (default 14)
        """
        from pyspark.sql.window import Window
        
        window_spec = Window.partitionBy("symbol").orderBy("timestamp")
        
        # Calculate price changes
        df_with_change = df.withColumn(
            "price_change",
            col("close") - lag("close", 1).over(window_spec)
        )
        
        # Separate gains and losses
        df_with_gains_losses = df_with_change \
            .withColumn("gain", when(col("price_change") > 0, col("price_change")).otherwise(0)) \
            .withColumn("loss", when(col("price_change") < 0, -col("price_change")).otherwise(0))
        
        # Calculate average gain and loss
        avg_window = window_spec.rowsBetween(-period+1, 0)
        df_with_avg = df_with_gains_losses \
            .withColumn("avg_gain", avg("gain").over(avg_window)) \
            .withColumn("avg_loss", avg("loss").over(avg_window))
        
        # Calculate RS and RSI
        df_with_rsi = df_with_avg \
            .withColumn("rs", col("avg_gain") / (col("avg_loss") + 0.000001)) \
            .withColumn("rsi", 100 - (100 / (1 + col("rs"))))
        
        return df_with_rsi
    
    def calculate_bollinger_bands(self, df, period: int = 20, std_dev: float = 2.0):
        """
        Calculate Bollinger Bands
        
        Args:
            df: DataFrame with price data
            period: Moving average period
            std_dev: Number of standard deviations
        """
        from pyspark.sql.window import Window
        
        window_spec = Window.partitionBy("symbol").orderBy("timestamp")
        bb_window = window_spec.rowsBetween(-period+1, 0)
        
        df_with_bb = df \
            .withColumn("bb_middle", avg("close").over(bb_window)) \
            .withColumn("bb_std", stddev("close").over(bb_window)) \
            .withColumn("bb_upper", col("bb_middle") + (std_dev * col("bb_std"))) \
            .withColumn("bb_lower", col("bb_middle") - (std_dev * col("bb_std"))) \
            .withColumn("bb_width", col("bb_upper") - col("bb_lower")) \
            .withColumn("bb_position", (col("close") - col("bb_lower")) / col("bb_width"))
        
        return df_with_bb
    
    def calculate_macd(self, df, fast: int = 12, slow: int = 26, signal: int = 9):
        """
        Calculate MACD (Moving Average Convergence Divergence)
        
        Args:
            df: DataFrame with price data
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
        """
        from pyspark.sql.window import Window
        
        window_spec = Window.partitionBy("symbol").orderBy("timestamp")
        
        # Calculate EMAs (approximation with SMA for streaming)
        fast_window = window_spec.rowsBetween(-fast+1, 0)
        slow_window = window_spec.rowsBetween(-slow+1, 0)
        signal_window = window_spec.rowsBetween(-signal+1, 0)
        
        df_with_ema = df \
            .withColumn("ema_fast", avg("close").over(fast_window)) \
            .withColumn("ema_slow", avg("close").over(slow_window))
        
        df_with_macd = df_with_ema \
            .withColumn("macd", col("ema_fast") - col("ema_slow")) \
            .withColumn("macd_signal", avg("macd").over(signal_window)) \
            .withColumn("macd_histogram", col("macd") - col("macd_signal"))
        
        return df_with_macd
    
    def process_and_write_features(self):
        """Main processing pipeline"""
        
        self.logger.info("Starting feature extraction pipeline")
        
        # Create input stream
        price_stream = self.create_stock_price_stream()
        
        # Calculate all indicators
        features = self.calculate_technical_indicators(price_stream)
        features = self.calculate_rsi(features)
        features = self.calculate_bollinger_bands(features)
        features = self.calculate_macd(features)
        
        # Add metadata
        features_with_meta = features \
            .withColumn("processing_time", current_timestamp()) \
            .withColumn("date", to_date(col("timestamp")))
        
        # Write to console (for debugging)
        query_console = features_with_meta \
            .writeStream \
            .outputMode("append") \
            .format("console") \
            .option("truncate", False) \
            .option("numRows", 5) \
            .start()
        
        # Write to Kafka (processed features topic)
        query_kafka = features_with_meta \
            .selectExpr("symbol as key", "to_json(struct(*)) as value") \
            .writeStream \
            .outputMode("append") \
            .format("kafka") \
            .option("kafka.bootstrap.servers", self.kafka_brokers) \
            .option("topic", "processed.features") \
            .option("checkpointLocation", f"{self.checkpoint_dir}/kafka") \
            .start()
        
        # Write to Parquet (for batch analysis)
        query_parquet = features_with_meta \
            .writeStream \
            .outputMode("append") \
            .format("parquet") \
            .option("path", "/data/processed_features") \
            .option("checkpointLocation", f"{self.checkpoint_dir}/parquet") \
            .partitionBy("date", "symbol") \
            .start()
        
        self.logger.info("Feature extraction pipeline started")
        
        # Wait for all queries
        query_console.awaitTermination()
        query_kafka.awaitTermination()
        query_parquet.awaitTermination()
    
    def stop(self):
        """Stop Spark session"""
        self.logger.info("Stopping Spark session")
        self.spark.stop()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run job
    job = FeatureExtractionJob(
        kafka_brokers="localhost:9092",
        checkpoint_dir="/tmp/spark-checkpoints"
    )
    
    try:
        job.process_and_write_features()
    except KeyboardInterrupt:
        job.stop()
