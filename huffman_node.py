# huffman coding
class HuffmanNode:
    def __init__(self, content: tuple, probability: float, left_child=None, right_child=None) -> None:
        """Node class used to build a Huffman tree.

        Args:
            content (tuple[Any]): hashable content of the node.
            probability (float): probability associated with the node's content.
            left_child (HuffmanNode, optional): left child node. Defaults to None.
            right_child (HuffmanNode, optional): right child node. Defaults to None.
        """
        self.content = content
        self.probability = probability
        self.left_child = left_child
        self.right_child = right_child
    
    def __str__(self, indent=0):
        """Recursive method that is called when a print() call is issued on the object.

        Args:
            indent (int, optional): number of spaces to display the node content. Defaults to 0.

        Returns:
            str: the string of the tree from this symbol.
        """
        str_content = f"{self.content}"
        indent_str = "| " * indent
        str_left =  "" if self.left_child is None else "\n" +  indent_str + f"├ 0: {self.left_child.__str__(indent=indent + 1)}"
        str_right = "" if self.right_child is None else "\n" +  indent_str + f"├ 1: {self.right_child.__str__(indent=indent + 1)}"
        return str_content + str_left + str_right
    
    def codes(self, base_code=""):
        """Return the strings of binary code for each leaves in the tree.

        Args:
            base_code (str, optional): _description_. Defaults to "".

        Returns:
            _type_: _description_
        """
        if self.left_child is None and self.right_child is None:
            c = {self.content: base_code}
        else:
            left_codes = {} if self.left_child is None else self.left_child.codes(base_code + "0")
            right_codes = {} if self.right_child is None else self.right_child.codes(base_code + "1")
            c = dict(sorted({**left_codes, **right_codes}.items()))
        return c

    