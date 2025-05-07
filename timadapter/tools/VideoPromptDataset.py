from torch.utils.data import Dataset, DataLoader
import json
import os
from timadapter.tools.process_video_frames import process_video_frames
import random
from torchvision import transforms
from torch.utils.data.sampler import BatchSampler, Sampler



def load_prompts_videos(prompt_path, start_idx=None, end_idx=None):
    prompt_list = []
    video_list  = []
    if prompt_path.endswith(".jsonl"):
        with open(prompt_path, "r") as f:
            for line in f.readlines():
                item = json.loads(line)
                prompt = item["description"]["qwen2-VL-72B-detail"]
                video_path = item["video_path"]
                if os.path.exists(video_path):
                    prompt_list.append(prompt)
                    video_list.append(video_path)
    elif prompt_path.endswith(".json"):
        with open(prompt_path, 'r') as f:
            json_data_list = json.load(f)
            for item in json_data_list:
                prompt_list.append(item["prompt"])
                video_list.append(item["image"])
    else:
        raise ValueError("The prompt_path must end with .txt or .jsonl.")
    prompt_list = prompt_list[start_idx:end_idx]
    video_list  = video_list[start_idx:end_idx]

    return prompt_list, video_list


class VideoPromptDataset(Dataset):
    def __init__(self, prompt_list, video_list, image_size=[224, 224]):
        self.prompt_list = prompt_list
        self.video_list  = video_list
        self.length      = len(self.prompt_list)
        self.image_size  = image_size # [height, width]
        self.to_tensor   = transforms.ToTensor()
        
    def __len__(self):
        return len(self.prompt_list)
    
    def __getitem__(self, idx):
        while True:
            sample = {}
            try:
                prompt     = self.prompt_list[idx]
                video_path = self.video_list[idx]

                image_frame = process_video_frames(video_path, target_size=self.image_size, adpative_size=False)
                if image_frame is not None:
                    sample["prompt"]      = prompt
                    sample["video_path"]  = video_path
                    sample["image_frame"] = self.to_tensor(image_frame) # 将 PIL.Image 转换为 Tensor
                    sample["data_type"]   = "video"
                
                if len(sample) > 0:
                    break
            except Exception as e:
                print(f"Error occurred while processing sample {idx}: {e}")
                idx = random.randint(0, self.length-1)

        return sample


class PromptVideoSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler: Sampler,
                 dataset: Dataset,
                 batch_size: int,
                 drop_last: bool = False
                ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

        # buckets for each aspect ratio
        self.bucket = {'image':[], 'video':[]}

    def __iter__(self):
        for idx in self.sampler:
            content_type = self.dataset[idx].get('data_type', 'video')
            self.bucket[content_type].append(idx)

            # yield a batch of indices in the same aspect ratio group
            if len(self.bucket['video']) == self.batch_size:
                bucket = self.bucket['video']
                yield bucket[:]
                del bucket[:]
            elif len(self.bucket['image']) == self.batch_size:
                bucket = self.bucket['image']
                yield bucket[:]
                del bucket[:]