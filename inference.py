import numpy as np
from text_model import SimpleVocab
import torch
import img_text_composition_models
from main import parse_opt, load_dataset
import tqdm
from tqdm import tqdm as tqdm
import os.path
import PIL
import torchvision
from pathlib import Path

def main():
    opt = parse_opt()
    trainset, testset = load_dataset(opt)
    # input_img = input()
    embed_dim = 512

    # print(trainset[0])
    texts = [t for t in trainset.get_all_texts()]
    print(len(texts))
    path = 'checkpoints/checkpoint_fashion200k.pth'
    device = torch.device('cpu')


    vocab = SimpleVocab()
    for text in texts:
        vocab.add_text_to_vocab(text)

    vocab_size = vocab.get_size()

    model = img_text_composition_models.TIRG(vocab_size, embed_dim=embed_dim)
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    texts = ['green linen-blend a-line dress']
    # x = vocab.encode_text(texts)

    in_dir = Path('./output/source')
    in_dir.mkdir(parents=True, exist_ok=True)

    input_img_path = Path(opt.dataset_path).joinpath('women/dresses/casual_and_day_dresses/51727804/51727804_0.jpeg')
    
    with open(input_img_path, 'rb') as f:
        img = PIL.Image.open(f)
        img.save(str(in_dir.joinpath(input_img_path.parts[-1])))
        img = img.convert('RGB')
        print(input_img_path, str(in_dir.joinpath(input_img_path.parts[-1])))
    
    transform=torchvision.transforms.Compose([
              torchvision.transforms.Resize(224),
              torchvision.transforms.CenterCrop(224),
              torchvision.transforms.ToTensor(),
              torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                               [0.229, 0.224, 0.225])
            ])
    img = transform(img).unsqueeze(0)
    # print(img.shape)
    # to tensor
    # img = torch.rand(1, 3, 256, 256)
    itexts = [vocab.encode_text(s) for s in texts]
    lengths = [len(t) for t in itexts]

    pitexts = torch.zeros((np.max(lengths), len(texts))).long()     # [T, B]
    for i in range(len(texts)):
      pitexts[:lengths[i], i] = torch.tensor(itexts[i])
    pitexts = pitexts.to(device)

    all_queries = model.compose_img_text(img, pitexts, lengths).data.cpu().numpy()

    # compute all image features
    if os.path.isfile('database.npy'):
        with open('database.npy', 'rb') as f:
            all_imgs = np.load(f)
    else:
        imgs = []
        all_imgs = []

        for i in tqdm(range(len(testset.imgs))):
            imgs += [testset.get_img(i)]
            if len(imgs) >= opt.batch_size or i == len(testset.imgs) - 1:
                if 'torch' not in str(type(imgs[0])):
                    imgs = [torch.from_numpy(d).float() for d in imgs]
                imgs = torch.stack(imgs).float()
                imgs = imgs.to(device)
                imgs = model.extract_img_feature(imgs).data.cpu().numpy()
                all_imgs += [imgs]
                imgs = []

        all_imgs = np.concatenate(all_imgs)
        with open('database.npy', 'wb') as f:
            np.save(f, all_imgs)


    # feature normalization
    for i in range(all_queries.shape[0]):
        all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
    for i in range(all_imgs.shape[0]):
        all_imgs[i, :] /= np.linalg.norm(all_imgs[i, :])

    # match test queries to target images, get nearest neighbors
    top_n = 5
    sims = all_queries.dot(all_imgs.T)
    sims = np.squeeze(sims)
    nn_result = np.argsort(-sims)[:top_n]
    
    out_dir = Path('./output/target')
    out_dir.mkdir(parents=True, exist_ok=True)

    for n in nn_result:
        file_path = Path(opt.dataset_path).joinpath(testset.imgs[n]['file_path'])
        print(file_path)
        with open(file_path, 'rb') as f:
            img = PIL.Image.open(f)
            img.save(str(out_dir.joinpath(file_path.parts[-1])))


    # if test_queries:
    #     for i, t in enumerate(test_queries):
    #         sims[i, t['source_img_id']] = -10e10  # remove query image
    # nn_result = [np.argsort(-sims[i, :])[:110] for i in range(sims.shape[0])]


if __name__ == '__main__':
    main()