from dataclasses import dataclass
from typing import Generic, TypeVar, Set, Iterable
from abc import ABC, abstractmethod


# Base class for QPL types
class QPLType(ABC):
    @abstractmethod
    def __repr__(self):
        raise NotImplementedError
    
    def __eq__(self, other):
        if not isinstance(other, QPLType):
            return False
        return repr(self) == repr(other)
    
    def __hash__(self):
        return hash(repr(self))

# Primitive types
@dataclass(unsafe_hash=True)
class Number(QPLType):
    def __repr__(self):
        return f"Number"


@dataclass(unsafe_hash=True)
class Entity(QPLType):
    name: str

    def __repr__(self):
        return self.name


# Entity composition types
E = TypeVar("E", bound=Entity)

@dataclass(unsafe_hash=True)
class Partial(QPLType, Generic[E]):
    entity: E

    def __repr__(self):
        return f"Partial[{self.entity}]"

@dataclass(unsafe_hash=True)
class Reduced(QPLType, Generic[E]):
    entity: E

    def __repr__(self):
        return f"Reduced[{self.entity}]"


# Type composition types
T = TypeVar("T", bound=QPLType)

@dataclass(unsafe_hash=True)
class TypeList(QPLType, Generic[T]):
    type: T

    def __repr__(self):
        return f"List[{self.type}]"


class Union(QPLType, Generic[T]):
    types: Set[T]

    def __init__(self, types: Iterable[T]):
        flattened: Set[T] = set()
        for t in types:
            if isinstance(t, Union):
                flattened.update(t.types)
            else:
                flattened.add(t)
        
        self.types = set()
        for t in flattened:
            if isinstance(t, Partial) and t.entity in flattened:
                # Entity -> Partial[Entity]
                continue
            self.types.add(t)

    def __repr__(self):
        return f"Union[{', '.join(map(str, self.types))}]"
