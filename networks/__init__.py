from networks.model import AdversarialModel

all_models = {
    'adversarial_model': AdversarialModel
}


def get_model(name):
    return all_models[name]