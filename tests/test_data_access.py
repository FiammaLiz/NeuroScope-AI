import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest
from pydantic import Field

from neuroscope.analysis.contracts import (
    DataReference,
    OperationDefinition,
    OperationModel,
    SampleInterval,
)
from neuroscope.analysis.operations import (
    ApprovalRequiredError,
    OperationRunner,
)
from neuroscope.analysis.registry import OperationRegistry
from neuroscope.data_access import DatasetAccessError, DatasetStore
from neuroscope.domain import DatasetProfile
from neuroscope.exploration import DatasetExploration, MetadataStatus


def _exploration(values: np.ndarray, *, channel_axis: int | None = None):
    profile = DatasetProfile.from_array(
        Path("synthetic.npy"), values, channel_axis=channel_axis
    )
    return DatasetExploration(values, profile)


def _reference(registration, channels=(0,), start=0, stop=1, *, component_id=None):
    return registration.reference(
        channels,
        SampleInterval(start=start, stop=stop),
        component_id=component_id,
    )


def test_register_and_resolve_one_dimensional_data_as_one_channel():
    values = np.arange(8, dtype=np.int16)
    before = values.copy()
    store = DatasetStore()
    registration = store.register(_exploration(values))
    reference = _reference(registration, start=2, stop=6)

    resolved = store.resolve(reference)

    assert registration.channel_count == 1
    assert registration.sample_count == 8
    assert resolved.reference == reference
    assert resolved.channels[0].channel_index == 0
    assert resolved.sample_interval == SampleInterval(start=2, stop=6)
    np.testing.assert_array_equal(resolved.channel(0).samples, [2, 3, 4, 5])
    assert resolved.channel(0).samples.dtype == values.dtype
    assert np.shares_memory(resolved.channel(0).samples, values)
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("channel_axis", [0, 1])
def test_two_dimensional_resolution_respects_confirmed_orientation(channel_axis):
    values = np.arange(20, dtype=np.float64).reshape(4, 5)
    exploration = _exploration(values, channel_axis=channel_axis)
    store = DatasetStore()
    registration = store.register(exploration)
    reference = _reference(registration, channels=(1, 3), start=1, stop=3)

    resolved = store.resolve(reference)

    assert registration.channel_count == values.shape[channel_axis]
    assert registration.sample_count == values.shape[1 - channel_axis]
    expected_channel_1 = values[1, 1:3] if channel_axis == 0 else values[1:3, 1]
    expected_channel_3 = values[3, 1:3] if channel_axis == 0 else values[1:3, 3]
    assert tuple(item.channel_index for item in resolved.channels) == (1, 3)
    np.testing.assert_array_equal(resolved.channel(1).samples, expected_channel_1)
    np.testing.assert_array_equal(resolved.channel(3).samples, expected_channel_3)
    assert all(np.shares_memory(item.samples, values) for item in resolved.channels)


def test_resolved_views_are_read_only_exact_and_do_not_transform_signal():
    values = np.arange(30, dtype=np.int16).reshape(3, 10)
    before = values.copy()
    store = DatasetStore()
    registration = store.register(_exploration(values, channel_axis=0))
    resolved = store.resolve(_reference(registration, channels=(2,), start=4, stop=7))
    view = resolved.channel(2).samples

    assert view.flags.writeable is False
    assert view.dtype == np.int16
    assert np.shares_memory(view, values)
    assert view.shape == (3,)
    np.testing.assert_array_equal(view, [24, 25, 26])
    with pytest.raises(ValueError):
        view[0] = -1
    np.testing.assert_array_equal(values, before)


def test_invalid_channels_and_sample_windows_fail_at_resolution():
    values = np.zeros((2, 5), dtype=np.float32)
    store = DatasetStore()
    registration = store.register(_exploration(values, channel_axis=0))

    with pytest.raises(DatasetAccessError, match="channel_index|channel"):
        store.resolve(_reference(registration, channels=(2,), stop=2))
    with pytest.raises(DatasetAccessError, match="exceeds"):
        store.resolve(_reference(registration, channels=(0,), start=3, stop=6))
    with pytest.raises(ValueError, match="non-empty"):
        SampleInterval(start=2, stop=2)
    with pytest.raises(ValueError, match="non-negative"):
        DataReference(
            dataset=registration.dataset,
            session_handle=registration.session_handle,
            channels=(-1,),
            sample_interval=SampleInterval(start=0, stop=1),
        )


def test_unknown_component_and_unknown_session_handle_are_rejected():
    values = np.arange(4)
    store = DatasetStore()
    registration = store.register(_exploration(values))
    unknown_handle_reference = DataReference(
        dataset=registration.dataset,
        session_handle=uuid4(),
        channels=(0,),
        sample_interval=SampleInterval(start=0, stop=1),
    )
    component_reference = _reference(registration, component_id="audio")

    with pytest.raises(DatasetAccessError, match="unknown or stale"):
        store.resolve(unknown_handle_reference)
    with pytest.raises(DatasetAccessError, match="component"):
        store.resolve(component_reference)


def test_unregistration_and_reregistration_never_reuse_old_references():
    first_values = np.arange(5)
    second_values = np.arange(5) + 100
    store = DatasetStore()
    first = store.register(_exploration(first_values))
    old_reference = _reference(first, stop=3)

    store.unregister(first.session_handle)
    second = store.register(_exploration(second_values))
    assert second.session_handle != first.session_handle
    with pytest.raises(DatasetAccessError, match="unknown or stale"):
        store.resolve(old_reference)
    np.testing.assert_array_equal(
        store.resolve(_reference(second, stop=3)).channel(0).samples,
        [100, 101, 102],
    )


def test_replace_invalidates_previous_registration_and_issues_new_handle():
    store = DatasetStore()
    first = store.register(_exploration(np.arange(5)))
    old_reference = _reference(first, stop=2)
    replacement = store.replace(
        first.session_handle,
        _exploration(np.arange(5) + 20),
    )

    assert replacement.session_handle != first.session_handle
    with pytest.raises(DatasetAccessError, match="unknown or stale"):
        store.resolve(old_reference)
    np.testing.assert_array_equal(
        store.resolve(_reference(replacement, stop=2)).channel(0).samples,
        [20, 21],
    )


class SignalInput(OperationModel):
    signal: DataReference


class SumParameters(OperationModel):
    scale: float = Field(default=1.0, strict=True)


class SumOutput(OperationModel):
    value: float = Field(strict=True)


class CountingStore(DatasetStore):
    def __init__(self):
        super().__init__()
        self.resolve_count = 0

    def resolve(self, reference):
        self.resolve_count += 1
        return super().resolve(reference)


def _signal_operation(calls):
    def execute(input_data, parameters, context, runtime):
        calls.append((input_data.signal, runtime.resolved_data))
        resolved = runtime.data_for(input_data.signal)
        selected = np.concatenate([item.samples for item in resolved.channels])
        assert not hasattr(runtime, "dataset_store")
        return SumOutput(value=float(selected.sum() * parameters.scale))

    return OperationDefinition(
        operation_id="neuroscope.test.window-sum",
        name="Test-only window sum",
        description="Exercises controlled data resolution without scientific meaning.",
        version="1.0.0",
        implementation_version="test-1",
        input_model=SignalInput,
        parameters_model=SumParameters,
        output_model=SumOutput,
        execute=execute,
    )


def _signal_runner(store, calls):
    registry = OperationRegistry()
    registry.register(_signal_operation(calls))
    return OperationRunner(registry, data_store=store)


def test_prepare_and_approval_do_not_resolve_or_execute_signal_data():
    values = np.arange(20, dtype=np.float64).reshape(2, 10)
    store = CountingStore()
    registration = store.register(_exploration(values, channel_axis=0))
    reference = _reference(registration, channels=(1,), start=3, stop=6)
    calls = []
    runner = _signal_runner(store, calls)
    prepared = runner.prepare(
        "neuroscope.test.window-sum",
        "1.0.0",
        SignalInput(signal=reference),
        SumParameters(scale=2.0),
    )

    assert store.resolve_count == 0
    assert calls == []
    assert prepared.input_data.signal == reference
    assert reference.model_dump_json() in prepared._input_json
    assert "[13.0,14.0,15.0]" not in prepared._input_json
    request_snapshot = prepared.to_snapshot()
    assert request_snapshot["input"]["signal"]["session_handle"] == str(
        registration.session_handle
    )
    json.dumps(request_snapshot)
    approved = runner.approve(prepared, approved_by="researcher")
    assert store.resolve_count == 0
    assert calls == []
    approval_snapshot = approved.to_snapshot()
    assert approval_snapshot["prepared"]["input"]["signal"]["channels"] == [1]
    assert "[13.0,14.0,15.0]" not in json.dumps(approval_snapshot)


def test_approval_is_bound_to_exact_selection_and_registration_handle():
    values = np.arange(20, dtype=np.float64).reshape(2, 10)
    store = DatasetStore()
    registration = store.register(_exploration(values, channel_axis=0))
    original_reference = _reference(registration, channels=(1,), start=3, stop=6)
    calls = []
    runner = _signal_runner(store, calls)
    prepared = runner.prepare(
        "neuroscope.test.window-sum",
        "1.0.0",
        SignalInput(signal=original_reference),
        SumParameters(),
    )
    approval = runner.approve(prepared, approved_by="researcher")
    changed_window = _reference(
        registration, channels=(1,), start=4, stop=7
    )
    changed_selection = runner.prepare(
        "neuroscope.test.window-sum",
        "1.0.0",
        SignalInput(signal=changed_window),
        SumParameters(),
    )
    second_registration = store.register(_exploration(values, channel_axis=0))
    changed_registration = runner.prepare(
        "neuroscope.test.window-sum",
        "1.0.0",
        SignalInput(
            signal=_reference(second_registration, channels=(1,), start=3, stop=6)
        ),
        SumParameters(),
    )

    assert not approval.authorizes(changed_selection)
    assert not approval.authorizes(changed_registration)
    assert calls == []


def test_data_references_require_a_store_before_operation_execution():
    values = np.arange(6)
    registration = DatasetStore().register(_exploration(values))
    reference = _reference(registration, start=1, stop=4)
    calls = []
    runner = _signal_runner(None, calls)
    prepared = runner.prepare(
        "neuroscope.test.window-sum",
        "1.0.0",
        SignalInput(signal=reference),
        SumParameters(),
    )
    approved = runner.approve(prepared, approved_by="researcher")

    record = runner.execute(approved)

    assert calls == []
    assert record.status.value == "failed"
    assert record.error.error_type == "DatasetAccessError"
    assert record.data_references == (reference,)


def test_approved_execution_receives_only_requested_read_only_window():
    values = np.arange(20, dtype=np.float64).reshape(2, 10)
    original = values.copy()
    store = CountingStore()
    registration = store.register(_exploration(values, channel_axis=0))
    reference = _reference(registration, channels=(1,), start=3, stop=6)
    calls = []
    runner = _signal_runner(store, calls)
    prepared = runner.prepare(
        "neuroscope.test.window-sum",
        "1.0.0",
        SignalInput(signal=reference),
        SumParameters(scale=2.0),
    )
    approval = runner.approve(prepared, approved_by="researcher")

    with pytest.raises(ApprovalRequiredError):
        runner.execute(prepared)
    assert store.resolve_count == 0
    record = runner.execute(approval)

    assert store.resolve_count == 1
    assert len(calls) == 1
    seen_reference, resolved_data = calls[0]
    assert seen_reference == reference
    assert len(resolved_data) == 1
    selected = resolved_data[0].channel(1).samples
    assert selected.shape == (3,)
    assert selected.flags.writeable is False
    np.testing.assert_array_equal(selected, [13.0, 14.0, 15.0])
    assert record.output == SumOutput(value=84.0)
    assert record.data_references == (reference,)
    assert record.input_data.signal == reference
    assert "13.0" not in record._input_json
    np.testing.assert_array_equal(values, original)


def test_invalidation_after_approval_prevents_operation_call_and_is_recorded():
    values = np.arange(8)
    store = CountingStore()
    registration = store.register(_exploration(values))
    reference = _reference(registration, start=1, stop=4)
    calls = []
    runner = _signal_runner(store, calls)
    prepared = runner.prepare(
        "neuroscope.test.window-sum",
        "1.0.0",
        SignalInput(signal=reference),
        SumParameters(),
    )
    approval = runner.approve(prepared, approved_by="researcher")
    store.unregister(registration.session_handle)

    record = runner.execute(approval)

    assert store.resolve_count == 1
    assert calls == []
    assert record.status.value == "failed"
    assert record.error.error_type == "DatasetAccessError"
    assert record.data_references == (reference,)
    assert record.provenance.approval_id == approval.approval_id


def test_data_reference_identity_is_checked_against_registration():
    store = DatasetStore()
    registration = store.register(_exploration(np.arange(5)))
    mismatched = DataReference(
        dataset=type(registration.dataset)(dataset_id="different-persistent-id"),
        session_handle=registration.session_handle,
        channels=(0,),
        sample_interval=SampleInterval(start=0, stop=2),
    )

    with pytest.raises(DatasetAccessError, match="persistent identity"):
        store.resolve(mismatched)


def test_registered_metadata_is_preserved_without_assuming_unknown_values():
    store = DatasetStore()
    registration = store.register(_exploration(np.arange(5)))
    resolved = store.resolve(_reference(registration, stop=2))

    assert resolved.metadata["sampling_rate_hz"].value is None
    assert resolved.metadata["sampling_rate_hz"].status is MetadataStatus.UNKNOWN
    assert resolved.metadata["units"].value is None
    assert resolved.metadata["units"].status is MetadataStatus.UNKNOWN
