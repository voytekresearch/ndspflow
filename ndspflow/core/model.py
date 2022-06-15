"""Models."""


class Model:
    """Model wrapper."""

    def __init__(self, model=None, nodes=None):
        """Initialize model."""
        self.model = model
        self.nodes = nodes

        self.model_self = None
        self.return_attrs = None


    def fit(self, model, *args, **kwargs):
        """Queue fit."""
        self.model = model
        self.nodes.append(['fit', model, args, kwargs])


    def run_fit(self, x_arr, y_arr, *args, **kwargs):
        """Execute fit."""
        if isinstance(self.return_attrs, str):
            self.return_attrs = [self.return_attrs]

        if x_arr is not None:
            self.model.fit(x_arr, y_arr, *args, **kwargs)
        else:
            self.model.fit(y_arr, *args, **kwargs)

        # Transfer attribute from model to self
        for attr in self.return_attrs:
            if attr in ['self', 'model_self']:
                setattr(self, 'model_self', self.model)

            if hasattr(self.model, attr):
                setattr(self, attr, getattr(self.model, attr))
