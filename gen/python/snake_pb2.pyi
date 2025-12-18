from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Point(_message.Message):
    __slots__ = ("x", "y")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: int
    y: int
    def __init__(self, x: _Optional[int] = ..., y: _Optional[int] = ...) -> None: ...

class Snake(_message.Message):
    __slots__ = ("id", "health", "body")
    ID_FIELD_NUMBER: _ClassVar[int]
    HEALTH_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    id: str
    health: int
    body: _containers.RepeatedCompositeFieldContainer[Point]
    def __init__(self, id: _Optional[str] = ..., health: _Optional[int] = ..., body: _Optional[_Iterable[_Union[Point, _Mapping]]] = ...) -> None: ...

class GameState(_message.Message):
    __slots__ = ("width", "height", "snakes", "food", "you_id", "turn")
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    SNAKES_FIELD_NUMBER: _ClassVar[int]
    FOOD_FIELD_NUMBER: _ClassVar[int]
    YOU_ID_FIELD_NUMBER: _ClassVar[int]
    TURN_FIELD_NUMBER: _ClassVar[int]
    width: int
    height: int
    snakes: _containers.RepeatedCompositeFieldContainer[Snake]
    food: _containers.RepeatedCompositeFieldContainer[Point]
    you_id: str
    turn: int
    def __init__(self, width: _Optional[int] = ..., height: _Optional[int] = ..., snakes: _Optional[_Iterable[_Union[Snake, _Mapping]]] = ..., food: _Optional[_Iterable[_Union[Point, _Mapping]]] = ..., you_id: _Optional[str] = ..., turn: _Optional[int] = ...) -> None: ...

class InferenceRequest(_message.Message):
    __slots__ = ("data", "shape")
    DATA_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    shape: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, data: _Optional[bytes] = ..., shape: _Optional[_Iterable[int]] = ...) -> None: ...

class InferenceResponse(_message.Message):
    __slots__ = ("policies", "values")
    POLICIES_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    policies: _containers.RepeatedScalarFieldContainer[float]
    values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, policies: _Optional[_Iterable[float]] = ..., values: _Optional[_Iterable[float]] = ...) -> None: ...

class BatchInferenceResponse(_message.Message):
    __slots__ = ("responses",)
    RESPONSES_FIELD_NUMBER: _ClassVar[int]
    responses: _containers.RepeatedCompositeFieldContainer[InferenceResponse]
    def __init__(self, responses: _Optional[_Iterable[_Union[InferenceResponse, _Mapping]]] = ...) -> None: ...

class TrainingExample(_message.Message):
    __slots__ = ("state_data", "policies", "values")
    STATE_DATA_FIELD_NUMBER: _ClassVar[int]
    POLICIES_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    state_data: bytes
    policies: _containers.RepeatedScalarFieldContainer[float]
    values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, state_data: _Optional[bytes] = ..., policies: _Optional[_Iterable[float]] = ..., values: _Optional[_Iterable[float]] = ...) -> None: ...

class TrainingData(_message.Message):
    __slots__ = ("examples",)
    EXAMPLES_FIELD_NUMBER: _ClassVar[int]
    examples: _containers.RepeatedCompositeFieldContainer[TrainingExample]
    def __init__(self, examples: _Optional[_Iterable[_Union[TrainingExample, _Mapping]]] = ...) -> None: ...
