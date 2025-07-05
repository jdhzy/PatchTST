import os
import subprocess
import time

def train_patchtst(
    root_path='../../../../forecast/model_data/',
    data_path='hawaii_sample_60.csv',
    model_id='hawaii_allstations_1440_2160',
    seq_len=1440,
    label_len=336,
    pred_len=2160,
    patch_len=24,
    stride=12,
    d_model=64,
    e_layers=2,
    features='M',
    target='WVHT',
    train_epochs=5,
    itr=1,
    resume=True,
    checkpoints='./checkpoints/',
    batch_size=128,
    use_gpu=True
):
    cmd = [
        'python', 'run_longExp.py',
        '--is_training', '1',
        '--root_path', root_path,
        '--data_path', data_path,
        '--model_id', f'{model_id}_{seq_len}_{pred_len}',
        '--model', 'PatchTST',
        '--data', 'custom',
        '--features', features,
        '--target', target,
        '--seq_len', str(seq_len),
        '--label_len', str(label_len),
        '--pred_len', str(pred_len),
        '--patch_len', str(patch_len),
        '--stride', str(stride),
        '--e_layers', str(e_layers),
        '--d_model', str(d_model),
        '--train_epochs', str(train_epochs),
        '--des', 'exp_hawaii_full',
        '--itr', str(itr),
        '--loss', 'mse',
        '--dropout', '0.05',
        '--lradj', 'type3',
        '--batch_size', str(batch_size),
        '--checkpoints', checkpoints
    ]

    if resume:
        cmd.append('--resume')
    if not use_gpu:
        cmd += ['--use_gpu', 'False']

    print("Running command:\n", " ".join(cmd))
    start = time.time()
    subprocess.run(cmd)
    print(f"Finished in {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    train_patchtst()