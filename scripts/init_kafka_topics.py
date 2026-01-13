"""
Initialize Kafka topics for the system
"""
import logging
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import TopicAlreadyExistsError
import yaml


def load_config(config_path: str = "config/kafka_config.yaml"):
    """Load Kafka configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_topics(config):
    """Create all required Kafka topics"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Connect to Kafka
    admin_client = KafkaAdminClient(
        bootstrap_servers=config['brokers'],
        client_id='topic-creator'
    )
    
    # Get topic configuration
    topic_config = config['topic_config']
    topics_dict = config['topics']
    
    # Create NewTopic objects
    topics_to_create = []
    for topic_name in topics_dict.values():
        new_topic = NewTopic(
            name=topic_name,
            num_partitions=topic_config['num_partitions'],
            replication_factor=topic_config['replication_factor'],
            topic_configs={
                'retention.ms': str(topic_config['retention_ms']),
                'compression.type': topic_config['compression_type']
            }
        )
        topics_to_create.append(new_topic)
    
    # Create topics
    try:
        admin_client.create_topics(topics_to_create, validate_only=False)
        logger.info(f"Successfully created {len(topics_to_create)} topics")
        
        for topic in topics_to_create:
            logger.info(f"  - {topic.name}")
    
    except TopicAlreadyExistsError:
        logger.warning("Some topics already exist")
    
    except Exception as e:
        logger.error(f"Failed to create topics: {e}")
        raise
    
    finally:
        admin_client.close()


if __name__ == "__main__":
    config = load_config()
    create_topics(config)
