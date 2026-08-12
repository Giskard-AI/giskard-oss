from typing import Generic, TypeVar

import pytest
from giskard.core import Discriminated, discriminated_base
from pydantic import TypeAdapter, ValidationError


@discriminated_base
class Animal(Discriminated):
    """Base class for animals."""

    name: str


class Pet(Animal):
    """Base class for pets."""

    name: str


@Animal.register("tigger")
class Tigger(Pet):
    """A tigger."""

    stripes: int = 100


@Pet.register("cat")
class Cat(Pet):
    """A cat."""

    lives: int = 9


# Registering as Pet should work as well (same behavior as registering as Animal)
@Animal.register("dog")
class Dog(Pet):
    """A dog."""

    breed: str


@pytest.mark.parametrize(
    "animal,kind",
    [
        (Tigger(name="Tigger", stripes=100), "tigger"),
        (Dog(name="Buddy", breed="Labrador"), "dog"),
        (Cat(name="Whiskers", lives=9), "cat"),
    ],
)
def test_discriminated_base_registration(animal: Animal, kind: str):
    """Test that discriminated_base decorator registers the base class."""
    assert animal.kind == kind

    model_dump = animal.model_dump()
    assert model_dump["kind"] == kind
    assert model_dump["name"] == animal.name

    assert Animal.model_validate(model_dump) == animal
    assert Animal.model_validate_json(animal.model_dump_json()) == animal


def test_discriminated_invalid_kind():
    """Test that invalid kind raises an error."""
    data = {"kind": "elephant", "name": "Dumbo", "age": 10}
    with pytest.raises(
        ValueError, match=f"Kind elephant is not registered for class {Animal}"
    ):
        Animal.model_validate(data)


def test_discriminated_missing_kind():
    """Test that missing kind raises an error."""
    data = {"name": "Felix", "lives": 9}
    with pytest.raises(ValueError, match=f"Kind is not provided for class {Animal}"):
        Animal.model_validate(data)


T = TypeVar("T")


@discriminated_base
class GenericAnimal(Discriminated, Generic[T]):
    """Base class for generic animals."""

    name: str
    value: T


class GenericPet(GenericAnimal[T]):
    """Base class for generic pets."""


@GenericAnimal.register("dog")
class GenericDog(GenericPet[T]):
    """A dog."""

    breed: str


@GenericPet.register("cat")
class GenericCat(GenericPet[T]):
    """A cat."""

    lives: int


@GenericAnimal.register("tigger")
class GenericTigger(GenericPet[T]):
    """A tigger."""

    stripes: int


@pytest.mark.parametrize(
    "animal,kind",
    [
        (GenericDog(name="Buddy", value=100, breed="Labrador"), "dog"),
        (GenericCat(name="Whiskers", value="Meow", lives=9), "cat"),
        (GenericTigger(name="Tigger", value=1.0, stripes=100), "tigger"),
    ],
)
def test_discriminated_generic_base_registration(animal: GenericAnimal[T], kind: str):
    """Test that discriminated_base decorator registers the base class."""
    assert animal.kind == kind

    model_dump = animal.model_dump()
    assert model_dump["kind"] == kind
    assert model_dump["name"] == animal.name
    assert model_dump["value"] == animal.value

    assert GenericAnimal.model_validate(model_dump) == animal
    assert GenericAnimal.model_validate_json(animal.model_dump_json()) == animal


def test_discriminated_generic_with_concrete_type():
    """Test that discriminated_base decorator registers the base class."""
    dog = GenericDog(name="Buddy", value=100, breed="Labrador")
    model_dump = dog.model_dump()
    assert GenericAnimal[int].model_validate(model_dump) == dog


def test_complex_type_adapter():
    type_adapter = TypeAdapter(GenericAnimal | int)

    assert type_adapter.validate_python(1) == 1
    assert (
        type_adapter.validate_python("1") == 1
    )  # Pydantic will try to convert the string to an int
    with pytest.raises(ValidationError):
        type_adapter.validate_python("Not a number")

    dog = GenericDog(name="Buddy", value=100, breed="Labrador")
    assert type_adapter.validate_python(dog.model_dump()) == dog
    assert type_adapter.validate_python(dog) == dog


def test_type_adpter_with_dict():
    type_adapter = TypeAdapter(GenericAnimal | dict)

    assert type_adapter.validate_python({"test": 1}) == {
        "test": 1
    }  # No kind, will be parsed as dict
    assert type_adapter.validate_python({"kind": {"test": 1}}) == {
        "kind": {"test": 1}
    }  # Kind is not a string, will be parsed as dict
    assert type_adapter.validate_python({"kind": "dog"}) == {
        "kind": "dog"
    }  # Kind is a string, but will fail to parse as the corresponding class
    assert type_adapter.validate_python({"kind": "rabbit"}) == {
        "kind": "rabbit"
    }  # Kind is a string, but not registered for any class


@discriminated_base
class Furniture(Discriminated):
    """Base class for animals."""

    name: str


@Furniture.register("chair")
class Chair(Furniture):
    """A chair."""

    color: str


def test_type_adpter_with_multiple_discriminated():
    type_adapter = TypeAdapter(GenericAnimal | Furniture)

    assert type_adapter.validate_python(
        {"kind": "chair", "name": "Chair", "color": "Red"}
    ) == Chair(name="Chair", color="Red")
    assert type_adapter.validate_python(
        {"kind": "dog", "name": "Buddy", "value": 100, "breed": "Labrador"}
    ) == GenericDog(name="Buddy", value=100, breed="Labrador")


def test_type_adpter_with_multiple_discriminated_kind_conflicts():
    animal_first_adapter = TypeAdapter(GenericAnimal | Furniture)
    furniture_first_adapter = TypeAdapter(Furniture | GenericAnimal)

    @Furniture.register("dog")
    class WronglyRegisteredDog(Furniture):
        """A dog."""

        breed: str

    assert animal_first_adapter.validate_python(
        {"kind": "chair", "name": "Chair", "color": "Red"}
    ) == Chair(name="Chair", color="Red")
    assert animal_first_adapter.validate_python(
        {"kind": "dog", "name": "Buddy", "value": 100, "breed": "Labrador"}
    ) == GenericDog(name="Buddy", value=100, breed="Labrador")

    assert furniture_first_adapter.validate_python(
        {"kind": "chair", "name": "Chair", "color": "Red"}
    ) == Chair(name="Chair", color="Red")
    assert furniture_first_adapter.validate_python(
        {"kind": "dog", "name": "Buddy", "value": 100, "breed": "Labrador"}
    ) == WronglyRegisteredDog(name="Buddy", breed="Labrador")
