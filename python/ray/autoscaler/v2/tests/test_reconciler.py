# coding: utf-8
import os
import sys
import unittest

import pytest  # noqa

from ray._private.test_utils import load_test_config
from ray.autoscaler._private.event_summarizer import EventSummarizer
from ray.autoscaler._private.node_launcher import BaseNodeLauncher
from ray.autoscaler._private.node_provider_availability_tracker import (
    NodeProviderAvailabilityTracker,
)
from ray.autoscaler.v2.instance_manager.config import AutoscalingConfig
from ray.autoscaler.v2.instance_manager.instance_storage import InstanceStorage
from ray.autoscaler.v2.instance_manager.node_provider import NodeProviderAdapter
from ray.autoscaler.v2.instance_manager.reconciler import RayStateReconciler
from ray.autoscaler.v2.instance_manager.storage import InMemoryStorage
from ray.autoscaler.v2.instance_manager.subscribers.reconciler import InstanceReconciler
from ray.autoscaler.v2.tests.util import FakeCounter, create_instance
from ray.core.generated.autoscaler_pb2 import (
    ClusterResourceState,
    NodeState,
    NodeStatus,
)
from ray.core.generated.instance_manager_pb2 import Instance
from ray.tests.autoscaler_test_utils import MockProvider


class InstanceReconcilerTest(unittest.TestCase):
    def setUp(self):
        self.base_provider = MockProvider()
        self.availability_tracker = NodeProviderAvailabilityTracker()
        self.node_launcher = BaseNodeLauncher(
            self.base_provider,
            FakeCounter(),
            EventSummarizer(),
            self.availability_tracker,
        )
        self.config = AutoscalingConfig(
            load_test_config("test_ray_complex.yaml"), skip_content_hash=True
        )
        self.node_provider = NodeProviderAdapter(
            self.base_provider, self.node_launcher, self.config
        )

        self.instance_storage = InstanceStorage(
            cluster_id="test_cluster_id",
            storage=InMemoryStorage(),
        )
        self.reconciler = InstanceReconciler(
            instance_storage=self.instance_storage,
            node_provider=self.node_provider,
        )

    def tearDown(self):
        self.reconciler.shutdown()

    def test_handle_ray_failure(self):
        self.node_provider.create_nodes("worker_nodes1", 1)
        instance = Instance(
            instance_id="0",
            instance_type="worker_nodes1",
            cloud_instance_id="0",
            status=Instance.RAY_STOPPED,
        )
        assert not self.base_provider.is_terminated(instance.cloud_instance_id)
        success, verison = self.instance_storage.upsert_instance(instance)
        assert success
        instance.version = verison
        self.reconciler._handle_ray_failure([instance.instance_id])

        instances, _ = self.instance_storage.get_instances(
            instance_ids={instance.instance_id}
        )
        assert instances[instance.instance_id].status == Instance.STOPPING
        assert self.base_provider.is_terminated(instance.cloud_instance_id)

        # reconciler will detect the node is terminated and update the status.
        self.reconciler._reconcile_with_node_provider()
        instances, _ = self.instance_storage.get_instances(
            instance_ids={instance.instance_id}
        )
        assert instances[instance.instance_id].status == Instance.STOPPED


def test_ray_reconciler():

    # Empty ray state.
    ray_cluster_state = ClusterResourceState(node_states=[])

    im_instances = [
        create_instance("i-1", status=Instance.ALLOCATED, cloud_instance_id="c-1"),
    ]
    assert RayStateReconciler.reconcile(ray_cluster_state, im_instances) == {}

    # A newly running ray node with matching cloud instance id
    ray_cluster_state = ClusterResourceState(
        node_states=[
            NodeState(node_id=b"r-1", status=NodeStatus.RUNNING, instance_id="c-1"),
        ]
    )
    updates = RayStateReconciler.reconcile(ray_cluster_state, im_instances)
    assert list(updates.keys()) == ["i-1"]
    assert updates["i-1"].new_instance_status == Instance.RAY_RUNNING

    # A newly running ray node w/o matching cloud instance id.
    ray_cluster_state = ClusterResourceState(
        node_states=[
            NodeState(node_id=b"r-1", status=NodeStatus.RUNNING, instance_id="unknown"),
        ]
    )
    updates = RayStateReconciler.reconcile(ray_cluster_state, im_instances)
    assert list(updates.keys()) == []

    # A running ray node already reconciled.
    im_instances = [
        create_instance("i-1", status=Instance.RAY_RUNNING, cloud_instance_id="c-1"),
        create_instance(
            "i-2", status=Instance.STOPPING, cloud_instance_id="c-2"
        ),  # Already reconciled.
    ]
    ray_cluster_state = ClusterResourceState(
        node_states=[
            NodeState(node_id=b"r-1", status=NodeStatus.IDLE, instance_id="c-1"),
            NodeState(
                node_id=b"r-2", status=NodeStatus.IDLE, instance_id="c-2"
            ),  # Already being stopped
        ]
    )
    updates = RayStateReconciler.reconcile(ray_cluster_state, im_instances)
    assert list(updates.keys()) == []

    # draining ray nodes
    im_instances = [
        create_instance(
            "i-1", status=Instance.RAY_RUNNING, cloud_instance_id="c-1"
        ),  # To be reconciled.
        create_instance(
            "i-2", status=Instance.RAY_STOPPING, cloud_instance_id="c-2"
        ),  # Already reconciled.
        create_instance(
            "i-3", status=Instance.STOPPING, cloud_instance_id="c-3"
        ),  # Already reconciled.
    ]
    ray_cluster_state = ClusterResourceState(
        node_states=[
            NodeState(node_id=b"r-1", status=NodeStatus.DRAINING, instance_id="c-1"),
            NodeState(node_id=b"r-2", status=NodeStatus.DRAINING, instance_id="c-2"),
            NodeState(node_id=b"r-3", status=NodeStatus.DRAINING, instance_id="c-3"),
        ]
    )
    updates = RayStateReconciler.reconcile(ray_cluster_state, im_instances)
    assert list(updates.keys()) == ["i-1"]
    assert updates["i-1"].new_instance_status == Instance.RAY_STOPPING

    # dead ray nodes
    im_instances = [
        create_instance(
            "i-1", status=Instance.ALLOCATED, cloud_instance_id="c-1"
        ),  # To be reconciled.
        create_instance(
            "i-2", status=Instance.RAY_STOPPING, cloud_instance_id="c-2"
        ),  # To be reconciled.
        create_instance(
            "i-3", status=Instance.STOPPING, cloud_instance_id="c-3"
        ),  # Already reconciled.
    ]

    ray_cluster_state = ClusterResourceState(
        node_states=[
            NodeState(node_id=b"r-1", status=NodeStatus.DEAD, instance_id="c-1"),
            NodeState(node_id=b"r-2", status=NodeStatus.DEAD, instance_id="c-2"),
            NodeState(node_id=b"r-3", status=NodeStatus.DEAD, instance_id="c-3"),
        ]
    )
    updates = RayStateReconciler.reconcile(ray_cluster_state, im_instances)
    assert list(updates.keys()) == ["i-1", "i-2"]
    assert updates["i-1"].new_instance_status == Instance.RAY_STOPPED
    assert updates["i-2"].new_instance_status == Instance.RAY_STOPPED

    # Unknown ray node status - no action.
    im_instances = [
        create_instance(
            "i-1", status=Instance.ALLOCATED, cloud_instance_id="c-1"
        ),  # To be reconciled.
    ]
    ray_cluster_state = ClusterResourceState(
        node_states=[
            NodeState(node_id=b"r-1", status=NodeStatus.UNSPECIFIED, instance_id="c-1"),
        ]
    )
    updates = RayStateReconciler.reconcile(ray_cluster_state, im_instances)
    assert list(updates.keys()) == []


if __name__ == "__main__":
    if os.environ.get("PARALLEL_CI"):
        sys.exit(pytest.main(["-n", "auto", "--boxed", "-vs", __file__]))
    else:
        sys.exit(pytest.main(["-sv", __file__]))
