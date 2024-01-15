import torch
import torch.nn.functional as F

def fgsm(model, image):
    image.requires_grad = True
    output = model(image)
    loss = torch.nn.CrossEntropyLoss()(output, torch.ones_like(output))
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data
    # perturb image
    perturbed_image = image + data_grad.sign() * 0.1
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return image
def odin(model, image, feature_transform):
    perturbed_image = fgsm(model, image)
    return cross_entropy(model, perturbed_image).item()

def adv_jacobian(model, image, num_features=1):
    perturbed_image = fgsm(model, image)
    return jacobian(model, perturbed_image).item()

def jacobian(model, image, num_features=1):
    return torch.norm(
        torch.autograd.functional.jacobian(model, image),"fro").item()

def cross_entropy(model, image, num_features=1):
    return F.cross_entropy(model(image), torch.ones_like(model(image))).item()


def grad_magnitude(model, image, num_features=1):
    image.requires_grad = True
    output = model(image)
    loss = torch.nn.CrossEntropyLoss()(output, torch.ones_like(output))
    model.zero_grad()
    loss.backward()
    data_grad = image.grad.data
    return torch.norm(data_grad, "fro"  ).item()

def jjtsvd(model, image, num_features=1):
    jac = torch.autograd.functional.jacobian(model, image)
    jac = jac.flatten(-3).squeeze()
    jj = jac@jac.T
    return torch.svd(jj)[1][0].item()

def jtjsvd(model, image, num_features=1):
    jac = torch.autograd.functional.jacobian(model, image)
    jac = jac.flatten(-3).squeeze()
    jj = jac.T@jac
    return torch.svd(jj)[1][0].item()

def jjmag(model, image, num_features=1):
    jac = torch.autograd.functional.jacobian(model, image)
    jac = jac.flatten(-3).squeeze()
    jj = jac.T@jac
    return torch.norm(jj, "fro").item()
def condition_number(model, image, num_features=1):
    """
    https://books.google.no/books?id=JaPtxOytY7kC&q=978-0898713619&redir_esc=y#v=onepage&q=978-0898713619&f=false

    :return:
    """
    return torch.norm(
        torch.autograd.functional.jacobian(model, image),"fro").item() / (
        torch.norm(model(image), "fro").item()*torch.norm(image, "fro").item())