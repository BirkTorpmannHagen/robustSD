import torch
import torch.nn.functional as F


def jacobian(model, image):
    return torch.norm(
        torch.autograd.functional.jacobian(model, image),"fro").item()


def grad_magnitude(model, image):
    image.requires_grad = True
    output = model(image)
    loss = torch.nn.CrossEntropyLoss()(output, torch.ones_like(output))
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data
    return torch.norm(data_grad, "fro"  ).item()

def condition_number(model, image):
    """
    https://books.google.no/books?id=JaPtxOytY7kC&q=978-0898713619&redir_esc=y#v=onepage&q=978-0898713619&f=false

    :return:
    """
    return torch.norm(
        torch.autograd.functional.jacobian(model, image),"fro").item() / (
        torch.norm(model(image), "fro").item()*torch.norm(image, "fro").item())