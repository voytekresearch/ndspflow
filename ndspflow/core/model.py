"""Models."""


class Model:
    """Model wrapper."""

    def __init__(self, model, nodes=None, *args, **kwargs):
        """Initialize model."""
        self.model = model(*args, **kwargs)
        self.nodes = nodes
        self.return_attrs = None


    def fit(self, *args, **kwargs):
        """Queue fit."""
        self.nodes.append(['fit', args, kwargs])


    def run_fit(self, x_arr, y_arr, return_attrs, *args, **kwargs):
        """Execute fit."""
        self.return_attrs = [return_attrs] if isinstance(return_attrs, str) else return_attrs

        if x_arr is not None:
            self.model.model.fit(x_arr, y_arr, *args, *kwargs)
        else:
            self._model.model.fit(y_arr, *args, **kwargs)

        # Transfer attribute from model to self
        for attr in self.return_attrs:
            setattr(self, getattr(self.model.model, attr))
