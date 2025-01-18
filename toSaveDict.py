import torch
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tool to convert model to state dict')
    parser.add_argument('-m', '--model_path', default="./", type=str, help='path to where you save the model')
    parser.add_argument('-s', '--save_path', default="./", type=str, help='path where to save your model in state dict format')
    args = parser.parse_args()

    print(args.model_path + "  to  " + args.save_path)

    model = torch.load(args.model_path, weights_only=False)
    torch.save(model.state_dict(), args.save_path)
