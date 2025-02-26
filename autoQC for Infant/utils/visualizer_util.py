import time
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.profiler import profile, tensorboard_trace_handler


class Visualizers:
    def __init__(self, visual_root, savegraphs=True):
        self.writer = SummaryWriter(visual_root)
        self.sub_root = time.strftime('%Y年%m月%d日%H时%M分%S秒', time.localtime())
        self.savegraphs = savegraphs

    def vis_write(self, main_tag, tag_scalar_dict, global_step):
        self.writer.add_scalars(self.sub_root + '_{}'.format(main_tag), tag_scalar_dict, global_step)

    def vis_graph(self, model, input_to_model=None):
        if self.savegraphs:
            with self.writer as w:
                w.add_graph(model, input_to_model)
                self.savegraphs = False

    def vis_image(self, tag, img_tensor, epoch=None, step=None, formats='CHW'):
        if epoch is not None:
            self.writer.add_image(self.sub_root + f'_{tag}_{epoch}', img_tensor, global_step=step, dataformats=formats)
        else:
            self.writer.add_image(self.sub_root + f'_{tag}', img_tensor, global_step=step, dataformats=formats)

    def vis_images(self, tag, img_tensor, epoch=None, step=None, formats='NCHW'):
        if epoch is not None:
            self.writer.add_images(self.sub_root + f'_{tag}_{epoch}', img_tensor, global_step=step, dataformats=formats)
        else:
            self.writer.add_images(self.sub_root + f'_{tag}', img_tensor, global_step=step, dataformats=formats)

    def close_vis(self):
        self.writer.close()


def analysis_profile(config, model, save_dir):
    model.eval()
    input_size = [config['input_batch_size'], *config["input_size"]]
    inputs = torch.randn(input_size)

    with profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=tensorboard_trace_handler(f'{save_dir}/analysis'),
            profile_memory=True,
            record_shapes=True,
            with_stack=True
    ) as profiler:
        start = time.time()
        model(inputs)
        cost = time.time() - start
        print(f"predict_cost = {cost}s")
        profiler.step()

    print(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print(profiler.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
