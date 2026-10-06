"""
Trimmed-down pytorch_geometric: https://github.com/pyg-team/pytorch_geometric.
Copyright (c) 2023 PyG Team, MIT License (see NOTICE).

Local FlashCart modifications are marked with "# FlashCart:" comments.
"""

import re

import torch


def size_repr(key, item, indent=0):
    indent_str = " " * indent
    if torch.is_tensor(item) and item.dim() == 0:
        out = item.item()
    elif torch.is_tensor(item):
        out = str(list(item.size()))
    elif isinstance(item, list) or isinstance(item, tuple):
        out = str([len(item)])
    elif isinstance(item, dict):
        lines = [indent_str + size_repr(k, v, 2) for k, v in item.items()]
        out = "{\n" + ",\n".join(lines) + "\n" + indent_str + "}"
    elif isinstance(item, str):
        out = f'"{item}"'
    else:
        out = str(item)

    return f"{indent_str}{key}={out}"


class Data(object):
    """Represent one graph and its optional attributes.

    Args:
        x (Tensor, optional): Node feature matrix with shape
            :obj:`[num_nodes, num_node_features]`. (default: :obj:`None`)
        edge_index (LongTensor, optional): Graph connectivity in COO format with shape
            :obj:`[2, num_edges]`. (default: :obj:`None`)
        edge_attr (Tensor, optional): Edge feature matrix with shape
            :obj:`[num_edges, num_edge_features]`. (default: :obj:`None`)
        y (Tensor, optional): Graph or node targets with arbitrary shape. (default:
            :obj:`None`)
        pos (Tensor, optional): Node position matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        normal (Tensor, optional): Normal vector matrix with shape
            :obj:`[num_nodes, num_dimensions]`. (default: :obj:`None`)
        face (LongTensor, optional): Face adjacency matrix with shape
            :obj:`[3, num_faces]`. (default: :obj:`None`)
        **kwargs: Additional graph attributes stored on the instance.

    The data object is not restricted to these attributes and can be extended by any
    other additional data.

    Example::

        data = Data(x=x, edge_index=edge_index)
        data.train_idx = torch.tensor([...], dtype=torch.long)
        data.test_mask = torch.tensor([...], dtype=torch.bool)
    """

    def __init__(
        self,
        x=None,
        edge_index=None,
        edge_attr=None,
        y=None,
        pos=None,
        normal=None,
        face=None,
        **kwargs,
    ):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.pos = pos
        self.normal = normal
        self.face = face
        for key, item in kwargs.items():
            if key == "num_nodes":
                self.__num_nodes__ = item
            else:
                self[key] = item

        if edge_index is not None and edge_index.dtype != torch.long:
            raise ValueError(
                (f"Argument `edge_index` needs to be of type `torch.long` but " f"found type `{edge_index.dtype}`.")
            )

        if face is not None and face.dtype != torch.long:
            raise ValueError((f"Argument `face` needs to be of type `torch.long` but found " f"type `{face.dtype}`."))

    @classmethod
    def from_dict(cls, dictionary):
        """Construct a graph from a dictionary of attributes.

        Args:
            dictionary (dict): Attribute names and values to store without copying.

        Returns:
            Data: Graph holding the supplied attributes.
        """
        data = cls()

        for key, item in dictionary.items():
            data[key] = item

        return data

    def to_dict(self):
        """Return the stored, non-None attributes as a dictionary.

        Returns:
            dict: Attribute names and values. The values are not copied.
        """
        return {key: item for key, item in self}

    def __getitem__(self, key):
        r"""Gets the data of the attribute :obj:`key`."""
        return getattr(self, key, None)

    def __setitem__(self, key, value):
        """Sets the attribute :obj:`key` to :obj:`value`."""
        setattr(self, key, value)

    @property
    def keys(self):
        """Return the names of stored, non-None graph attributes.

        Returns:
            list[str]: Attribute names, excluding internal names that begin or end with
                a double underscore.
        """
        keys = [key for key in self.__dict__.keys() if self[key] is not None]
        keys = [key for key in keys if key[:2] != "__" and key[-2:] != "__"]
        return keys

    def __contains__(self, key):
        r"""Returns :obj:`True`, if the attribute :obj:`key` is present in the data."""
        return key in self.keys

    def __iter__(self):
        """Yield stored, non-None graph attributes in name order.

        Yields:
            tuple[str, Any]: Attribute name and its stored value.
        """
        for key in sorted(self.keys):
            yield key, self[key]

    def __call__(self, *keys):
        """Yield the requested graph attributes that are present and non-None.

        Args:
            *keys (str): Attribute names in the requested order. If omitted, use all
                stored attributes in name order.

        Yields:
            tuple[str, Any]: Attribute name and its stored value.
        """
        for key in sorted(self.keys) if not keys else keys:
            if key in self:
                yield key, self[key]

    def __cat_dim__(self, key, value):
        """Choose the concatenation dimension for a graph attribute.

        Names containing ``index`` or ``face`` concatenate along the final dimension.
        Other attributes concatenate along dimension zero. Subclasses can override this
        rule for attributes with a different layout.

        Args:
            key (str): Attribute name.
            value: Attribute value. Unused by this implementation.

        Returns:
            int: Concatenation dimension, either -1 or 0.
        """
        if bool(re.search("(index|face)", key)):
            return -1
        return 0

    def __inc__(self, key, value):
        """Choose the index offset applied when adding the next graph to a batch.

        Names containing ``index`` or ``face`` are offset by the number of nodes. Other
        attributes receive no offset. Subclasses can override this rule for other index
        conventions.

        Args:
            key (str): Attribute name.
            value: Attribute value. Unused by this implementation.

        Returns:
            int: Node count for index attributes, or zero for other attributes.
                The graph must have a known node count when an offset is required.
        """
        # Only `*index*` and `*face*` attributes should be cumulatively summed
        # up when creating batches.
        return self.num_nodes if bool(re.search("(index|face)", key)) else 0

    @property
    def num_nodes(self):
        """Return or set the number of nodes in the graph.

        An explicitly assigned value is used when available. Otherwise, the number
        is inferred from ``x``, ``pos``, ``normal``, ``batch``, ``adj``, or
        ``adj_t``, in that order. The node count is not inferred from ``edge_index``.

        Returns:
            int or None: Number of nodes, or None when it cannot be inferred.
        """
        if hasattr(self, "__num_nodes__"):
            return self.__num_nodes__
        for key, item in self("x", "pos", "normal", "batch"):
            return item.size(self.__cat_dim__(key, item))
        if hasattr(self, "adj"):
            return self.adj.size(0)
        if hasattr(self, "adj_t"):
            return self.adj_t.size(1)
        return None

    @num_nodes.setter
    def num_nodes(self, num_nodes):
        self.__num_nodes__ = num_nodes

    def __apply__(self, item, func):
        if torch.is_tensor(item):
            return func(item)
        elif isinstance(item, (tuple, list)):
            return [self.__apply__(v, func) for v in item]
        elif isinstance(item, dict):
            return {k: self.__apply__(v, func) for k, v in item.items()}
        else:
            return item

    def apply(self, func, *keys):
        """Transform tensor attributes in place and return this graph.

        The transformation also visits tensors inside lists, tuples, and dictionaries.
        Tuples are replaced by lists. If no keys are supplied, transform all stored
        attributes.

        Args:
            func (Callable): Function applied to each tensor.
            *keys (str): Attributes to transform.

        Returns:
            Data: This graph after its attributes have been updated.
        """
        for key, item in self(*keys):
            self[key] = self.__apply__(item, func)
        return self

    def contiguous(self, *keys):
        """Make tensor attributes contiguous in place.

        Args:
            *keys (str): Attributes to update. If omitted, update all tensor attributes,
                including those in nested containers.

        Returns:
            Data: This graph after its attributes have been updated.
        """
        return self.apply(lambda x: x.contiguous(), *keys)

    def to(self, device, *keys, **kwargs):
        """Convert tensor attributes in place.

        Args:
            device: Device or dtype passed as the first argument to ``Tensor.to``.
            *keys (str): Attributes to convert. If omitted, convert all tensor
                attributes, including those in nested containers.
            **kwargs: Additional arguments passed to ``Tensor.to``.

        Returns:
            Data: This graph after its attributes have been updated.
        """
        return self.apply(lambda x: x.to(device, **kwargs), *keys)

    def __repr__(self):
        cls = str(self.__class__.__name__)
        has_dict = any([isinstance(item, dict) for _, item in self])

        if not has_dict:
            info = [size_repr(key, item) for key, item in self]
            return "{}({})".format(cls, ", ".join(info))
        else:
            info = [size_repr(key, item, indent=2) for key, item in self]
            return "{}(\n{}\n)".format(cls, ",\n".join(info))
