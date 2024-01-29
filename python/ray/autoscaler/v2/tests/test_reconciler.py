# coding: utf-8
import os
import sys

import pytest  # noqa

from ray.autoscaler.v2.tests.util import create_instance
from ray.autoscaler.v2.instance_manager.reconciler import RayStateReconciler
from ray.core.generated.autoscaler_pb2 import (
    ClusterResourceState,
    NodeState,
    NodeStatus,
)
from ray.core.generated.instance_manager_pb2 import Instance


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
