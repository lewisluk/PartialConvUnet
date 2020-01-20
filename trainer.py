from scripts.PartialConvUnet import PartialConvUnet

if __name__ == '__main__':
    model = PartialConvUnet()
    model.init_network()
    model.restore()
    model.trainer()