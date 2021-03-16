import torch
import img_text_composition_models
from main import parse_opt, load_dataset

def main():
    opt = parse_opt()
    trainset, testset = load_dataset(opt)
    # input_img = input()
    embed_dim = 512

    # print(trainset[0])

    path = 'checkpoint_fashion200k.pth'
    device = torch.device('cpu')
    model = img_text_composition_models.TIRG(
            [t for t in trainset.get_all_texts()], embed_dim=embed_dim)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()


    input_text = 'green linen-blend a-line dress'
    input_img_path = '/media/huynhtruc0309/335274A52EB9F27F/hcmus/thesis/data/Fashion200k/women/dresses/casual_and_day_dresses/51727804/51727804_0.jpeg'
    model.compose_img_text()

if __name__ == '__main__':
    main()