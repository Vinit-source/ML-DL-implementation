import numpy as np
import MLlib
import MLlib.autograd as autograd


class Transpose(autograd.Function):

    @staticmethod
    def forward(ctx, a):

        if not (type(a).__name__ == 'Tensor'):
            raise Exception("The arg must be Tensor, got \
                {} instead".format(type(a).__name__))

        if not len(a.shape) == 2:
            raise Exception("Arg for Transpose must be 2D tensor, \
                got {}".format(a.shape))

        requires_grad = a.requires_grad

        b = MLlib.Tensor(a.data.T, requires_grad=requires_grad,
                         is_leaf=not requires_grad)

        return b

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        pass


class Reshape(autograd.Function):

    @staticmethod
    def forward(ctx, a, shape):

        if not type(a).__name__ == 'Tensor':
            raise Exception("Arg for Reshape must be tensor, got\
                         {}".format(type(a).__name__))

        requires_grad = a.requires_grad

        if requires_grad:
            ctx.shape = a.shape

        c = MLlib.Tensor(a.data.reshape(shape), requires_grad=requires_grad,
                         is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        pass


class Add(autograd.Function):

    @staticmethod
    def forward(ctx, a, b):

        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors, got \
                {}, {} instead".format(type(a).__name__, type(b).__name__))

        requires_grad = a.requires_grad or b.requires_grad

        if requires_grad:
            ctx.save_for_backward(a, b)

        c = MLlib.Tensor(a.data + b.data, requires_grad=requires_grad,
                         is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        pass


class Sub(autograd.Function):

    @staticmethod
    def forward(ctx, a, b):

        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors, got \
                {}, {} instead".format(type(a).__name__, type(b).__name__))

        requires_grad = a.requires_grad or b.requires_grad

        if requires_grad:
            ctx.save_for_backward(a, b)

        c = MLlib.Tensor(a.data - b.data, requires_grad=requires_grad,
                         is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        pass


class Mul(autograd.Function):

    @staticmethod
    def forward(ctx, a, b):

        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors, got \
                {}, {} instead".format(type(a).__name__, type(b).__name__))

        requires_grad = a.requires_grad or b.requires_grad

        if requires_grad:
            ctx.save_for_backward(a, b)

        c = MLlib.Tensor(a.data * b.data, requires_grad=requires_grad,
                         is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        pass


class Div(autograd.Function):
    @staticmethod
    def forward(ctx, a, b):

        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors, got \
                {}, {} instead".format(type(a).__name__, type(b).__name__))

        requires_grad = a.requires_grad or b.requires_grad

        if requires_grad:
            ctx.save_for_backward(a, b)

        c = MLlib.Tensor(a.data / b.data, requires_grad=requires_grad,
                         is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        pass


class MatMul(autograd.Function):

    @staticmethod
    def forward(ctx, a, b):

        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors, got \
                {}, {} instead".format(type(a).__name__, type(b).__name__))

        requires_grad = a.requires_grad or b.requires_grad

        if requires_grad:
            ctx.save_for_backward(a, b)

        c = MLlib.Tensor(np.matmul(a.data, b.data),
                         requires_grad=requires_grad,
                         is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        pass


class Pow(autograd.Function):

    @staticmethod
    def forward(ctx, a, b):

        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors, got \
                {}, {} instead".format(type(a).__name__, type(b).__name__))

        requires_grad = a.requires_grad or b.requires_grad

        if requires_grad:
            ctx.save_for_backward(a, b)

        c = MLlib.Tensor(np.power(a.data, b.data), requires_grad=requires_grad,
                         is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        pass


class Dot(autograd.Function):

    @staticmethod
    def forward(ctx, a, b):

        if not (type(a).__name__ == 'Tensor' and type(b).__name__ == 'Tensor'):
            raise Exception("Both args must be Tensors, got \
                {}, {} instead".format(type(a).__name__, type(b).__name__))

        requires_grad = a.requires_grad or b.requires_grad

        if requires_grad:
            ctx.save_for_backward(a, b)

        c = MLlib.Tensor(np.dot(a.data, b.data), requires_grad=requires_grad,
                         is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        pass


class Sum(autograd.Function):

    @staticmethod
    def forward(ctx, a, axis, keepdims):
        if not type(a).__name__ == 'Tensor':
            raise Exception("Only sum of tensor is supported")

        requires_grad = a.requires_grad

        if requires_grad:
            ctx.axis = axis
            ctx.shape = a.shape

            if axis is not None:
                ctx.len = a.shape[axis]

            ctx.keepdims = keepdims

        c = MLlib.Tensor(a.data.sum(axis=axis, keepdims=keepdims),
                         requires_grad=requires_grad,
                         is_leaf=not requires_grad)

        return c

    @staticmethod
    def backward(ctx, grad_output):
        # TODO
        pass
