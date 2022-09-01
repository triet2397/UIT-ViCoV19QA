class Checkpoint(object):
    """Checkpoint class"""
    @staticmethod
    def save(model,cell, path):
        """Save model using name"""
        name_tmp = model.name+"_"+ cell if model.name==RNN_NAME else model.name
        name = f'{name_tmp}.pt'
        torch.save(model.state_dict(), path+name)

    @staticmethod
    def load(model,path, name):
        """Load model using name"""
        model.load_state_dict(torch.load(path+name))
        return model