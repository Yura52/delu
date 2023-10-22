from typing import Literal

import numpy as np
import scipy.sparse
import sklearn.preprocessing
from sklearn.utils import check_random_state


class NoisyQuantileTransformer(sklearn.preprocessing.QuantileTransformer):
    """Like `sklearn.preprocessing.QuantileTransformer`, but more robust to features with few distinct values.

    `NoisyQuantileTransformer` aims at producing nicer outputs than
    `~sklearn.preprocessing.QuantileTransformer` for features with few distinct values
    (see the example below). The technical differences with
    `~sklearn.preprocessing.QuantileTransformer` are as follows:

    #. `NoisyQuantileTransformer` applies noise (typically, of a low magnitude) to the
       training data during the ``.fit(...)`` step to deduplicate equal values.
       Crucially, the noise IS NOT applied to the input data during
       the ``.transform(...)`` step.

    #. The default hyperparameters are adjusted for training deep learning models.

    **Usage**

    Let's compare to what new values the original values are mapped
    under different transformations. The common setup:

    >>> import sklearn.preprocessing
    >>> import delu.tabular as tab
    >>>
    >>> def compare_unique_values(one_feature_column):
    ...     column = np.array(one_feature_column)
    ...     train_size = len(column)
    ...
    ...     kwargs = {
    ...         'n_quantiles': min(train_size // 30, 1000),
    ...         'output_distribution': 'normal',
    ...         'subsample': 10**7,
    ...         'random_state': 0,
    ...     }
    ...     for name, transformer in [
    ...         (
    ...             'Raw',
    ...             None,
    ...         ),
    ...         (
    ...             'StandardScaler',
    ...             sklearn.preprocessing.StandardScaler(),
    ...         ),
    ...         (
    ...             'QuantileTransformer',
    ...             sklearn.preprocessing.QuantileTransformer(**kwargs),
    ...         ),
    ...         (
    ...             'NoisyQuantileTransformer',
    ...             tab.preprocessing.NoisyQuantileTransformer(**kwargs),
    ...         ),
    ...     ]:
    ...         transformed_column = column if transformer is None else (
    ...             transformer.fit_transform(column[:, None])[:, 0]
    ...         )
    ...         unique_values = np.unique(transformed_column)
    ...         print(f'{name:<24} {np.round(unique_values, 3).tolist()}')

    Let's consider a feature with two equally popular values and a large outlier.
    As expected, ``StandardScaler`` is affected by the outlier,
    and ``QuantileTransformer`` spreads the very different values uniformly.
    ``NoisyQuantileTransformer`` avoids both issues:

    >>> compare_unique_values([0.0] * 500 + [1.0] * 500 + [1e6])
    Raw                      [0.0, 1.0, 1000000.0]
    StandardScaler           [-0.032, -0.032, 31.623]
    QuantileTransformer      [-5.199, 0.626, 5.199]
    NoisyQuantileTransformer [-0.649, 0.734, 5.199]

    Let's consider a feature with two equally popular values.
    ``QuantileTransformer`` produces extreme values,
    which is not the case for ``NoisyQuantileTransformer``.
    Due to the nature of ``NoisyQuantileTransformer``,
    it produces slightly non-symmetrical results.

    >>> compare_unique_values([1.0] * 500 + [200.0] * 500)
    Raw                      [1.0, 200.0]
    StandardScaler           [-1.0, 1.0]
    QuantileTransformer      [-5.199, 5.199]
    NoisyQuantileTransformer [-0.648, 0.736]
    """  # noqa: E501

    def __init__(
        self,
        *,
        n_quantiles: int,
        output_distribution: Literal['normal', 'uniform'] = 'normal',
        subsample: int = 10_000_000,
        noise_hint: float = 1e-3,
        **kwargs,
    ) -> None:
        """
        Args:
            n_quantiles: the argument for `sklearn.preprocessing.QuantileTransformer`.
                If unsure, set to ``min(train_size // 30, 1000)``, where ``train_size``
                is the expected dataset size on the ``.fit(...)``
                step (e.g. ``len(X_train)``).
            output_distribution: the argument for
                `sklearn.preprocessing.QuantileTransformer`.
            subsample: the argument for `sklearn.preprocessing.QuantileTransformer`.
            noise_hint: for a given feature with the standard deviation ``std``, defines
                the noise standard deviation as ``min(noise_hint, noise_hint * std)``.
            kwargs: other arguments for `sklearn.preprocessing.QuantileTransformer`.
        """
        if noise_hint <= 0.0:
            raise ValueError(
                f"noise_hint must be positive. The provided value: {noise_hint=}"
            )
        super().__init__(
            n_quantiles=n_quantiles,
            output_distribution=output_distribution,
            subsample=subsample,
            **kwargs,
        )
        self.noise_hint = noise_hint

    def _fit_raise_bad_type(self, X):
        raise ValueError(
            'X must be either `numpy.ndarray` or `pandas.DataFrame`.'
            f' The provided value: {type(X)=}'
        )

    def fit(self, X, y=None):
        """Apply noise to ``X`` and forward the call to `sklearn.preprocessing.QuantileTransformer.fit`"""  # noqa: E501

        # NOTE: Scikit-learn aims to preserve the container type, column names and
        # other properties of inputs. It means that, here, we must not change those
        # properties.

        if isinstance(X, np.ndarray):
            if scipy.sparse.issparse(X):
                raise ValueError(
                    'NoisyQuantileTransformer cannot be fitted on sparse arrays'
                )
            std = X.std(axis=0)
        else:
            try:
                import pandas
            except ImportError:
                self._fit_raise_bad_type(X)
            if not isinstance(X, pandas.DataFrame):
                self._fit_raise_bad_type(X)
            sparse_columns = []
            for column_name, column_values in X.items():
                if pandas.api.types.is_sparse(column_values):
                    sparse_columns.append(column_name)
            if sparse_columns:
                raise ValueError(
                    'NoisyQuantileTransformer cannot be fitted on sparse dataframes.'
                    f' Sparse column names: {sparse_columns}'
                )
            # ddof=0 for consistency with numpy.
            std = X.std(axis=0, ddof=0).values

        noise_std = np.minimum(self.noise_hint, self.noise_hint * std)
        noise = check_random_state(self.random_state).normal(0.0, noise_std, X.shape)
        return super().fit(X + noise, y)
