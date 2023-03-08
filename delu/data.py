"""Missing batteries from `torch.utils.data`."""

from typing import Any, Callable, Iterable, Optional, Tuple, TypeVar, Union

from torch.utils.data import Dataset

T = TypeVar('T')


class Enumerate(Dataset):
    """Make dataset return both indices and items.

    .. rubric:: Tutorial

    .. testcode::

        from torch.utils.data import DataLoader, TensorDataset
        X, y = torch.randn(9, 2), torch.randn(9)
        dataset = TensorDataset(X, y)
        for batch_idx, batch in DataLoader(Enumerate(dataset), batch_size=3):
            print(batch_idx)

    .. testoutput::

        tensor([0, 1, 2])
        tensor([3, 4, 5])
        tensor([6, 7, 8])
    """

    def __init__(self, dataset: Dataset) -> None:
        """Initialize self.

        Args:
            dataset
        """
        self._dataset = dataset

    @property
    def dataset(self) -> Dataset:
        """Access the underlying dataset.

        Returns:
            The dataset.
        """
        return self._dataset

    def __len__(self) -> int:
        """Get the length of the underlying dataset."""
        return len(self._dataset)  # type: ignore

    def __getitem__(self, index) -> Tuple[Any, Any]:
        """Return index and the corresponding item from the underlying dataset.

        Args:
            index
        Returns:
            (index, item)
        """
        return index, self._dataset[index]


class FnDataset(Dataset):
    """A thin wrapper around a loader function and its arguments.

    `FnDataset` allows to avoid implementing Dataset-classes (well, at least in simple
    cases). Below you can find the full tutorial and typical use cases, but here is a
    quick example:

    Without `FnDataset`::

        class ImagesList(Dataset):
            def __init__(self, filenames, transform):
                self.filenames = filenames
                self.transform = transform

            def __len__(self):
                return len(self.filenames)

            def __getitem__(self, index):
                return self.transform(Image.open(self.filenames[index]))

        dataset = ImagesList(filenames, transform)

    With `FnDataset`::

        dataset = FnDataset(Image.open, filenames, transform)

    .. rubric:: Tutorial

    With vanilla PyTorch, in order to create a dataset you have to inherit from
    `torch.utils.data.Dataset` and implement three methods:

    - ``__init__``
    - ``__len__``
    - ``__getitem__``

    With `FnDataset` the only thing you *may* need to implement is the ``fn``
    argument that will power ``__getitem__``. The easiest way to learn
    `FnDataset` is to go through examples below.

    A list of images::

        dataset = FnDataset(Image.open, filenames)
        # dataset[i] returns Image.open(filenames[i])

    A list of images that are cached after the first load::

        from functools import lru_cache
        dataset = FnDataset(lru_cache(None)(Image.open), filenames)

    `pathlib.Path` is very useful when you want to create a dataset that reads from
    files. For example::

        images_dir = Path(...)
        dataset = FnDataset(Image.open, images_dir.iterdir())

    If there are many files, but you need only those with specific extensions, use
    `pathlib.Path.glob`::

        dataset = FnDataset(Image.open, images_dir.glob(*.png))

    If there are many files in many subfolders, but you need only those with specific
    extensions and that satisfy some condition, use `pathlib.Path.rglob`::

        dataset = FnDataset(
            Image.open, (x for x in images_dir.rglob(*.png) if condition(x))
        )

    A segmentation dataset::

        image_filenames = ...
        gt_filenames = ...

        def get(i):
            return Image.open(image_filenames[i]), Image.open(gt_filenames[i])

        dataset = FnDataset(get, len(image_filenames))

    A dummy dataset that demonstrates that `FnDataset` is a very general thing:

    .. testcode::

        def f(x):
            return x * 10

        def g(x):
            return x * 2

        dataset = FnDataset(f, 3, g)
        # dataset[i] returns g(f(i))
        assert len(dataset) == 3
        assert dataset[0] == 0
        assert dataset[1] == 20
        assert dataset[2] == 40

    """

    def __init__(
        self,
        fn: Callable[..., T],
        args: Union[int, Iterable],
        transform: Optional[Callable[[T], Any]] = None,
    ) -> None:
        """Initialize self.

        Args:
            fn: the function that produces values based on arguments from ``args``
            args: arguments for ``fn``. If an iterable, but not a list, then is
                casted to a list. If an integer, then the behavior is the same as for
                ``list(range(args))``. The size of ``args`` defines the return
                value for `FnDataset.__len__`.
            transform: if presented, is applied to the return value of `fn` in
                `FnDataset.__getitem__`

        Examples:
            .. code-block::

                import PIL.Image as Image
                import torchvision.transforms as T

                dataset = FnDataset(Image.open, filenames, T.ToTensor())
        """
        self._fn = fn
        if isinstance(args, Iterable):
            if not isinstance(args, list):
                args = list(args)
        self._args = args
        self._transform = transform

    def __len__(self) -> int:
        """Get the dataset size.

        See `FnDataset` for details.

        Returns:
            size
        """
        return len(self._args) if isinstance(self._args, list) else self._args

    def __getitem__(self, index: int) -> Any:
        """Get value by index.

        See `FnDataset` for details.

        Args:
            index
        Returns:
            value
        Raises:
            IndexError: if ``index >= len(self)``
        """
        if isinstance(self._args, list):
            x = self._args[index]
        elif index < self._args:
            x = index
        else:
            raise IndexError(f'Index {index} is out of range')
        x = self._fn(x)
        return x if self._transform is None else self._transform(x)
