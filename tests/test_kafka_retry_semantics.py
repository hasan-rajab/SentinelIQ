from ingestion.kafka_consumer import _rewind_failed_message


class FakeConsumer:
    def __init__(self):
        self.partition = None

    def seek(self, partition):
        self.partition = partition


class FakeMessage:
    def topic(self):
        return "sentineliq.metrics"

    def partition(self):
        return 3

    def offset(self):
        return 42


def test_failed_forward_rewinds_same_partition_and_offset():
    consumer = FakeConsumer()
    _rewind_failed_message(consumer, FakeMessage())

    assert consumer.partition.topic == "sentineliq.metrics"
    assert consumer.partition.partition == 3
    assert consumer.partition.offset == 42
