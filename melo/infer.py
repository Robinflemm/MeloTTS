import os
import click
from melo.api import TTS

@click.command()
@click.option('--ckpt_path', '-m', type=str, required=True, help="Path to the checkpoint file")
@click.option('--text', '-t', type=str, required=True, help="Text to convert to speech")
@click.option('--language', '-l', type=str, default="EN", help="Language of the model (default is 'EN')")
@click.option('--output_dir', '-o', type=str, default="outputs", help="Directory to save the output audio files")
def main(ckpt_path, text, language, output_dir):
    """
    Converts the provided text to speech using the specified TTS model checkpoint.
    
    Args:
        ckpt_path (str): Path to the model checkpoint file.
        text (str): Text to convert to speech.
        language (str): Language of the TTS model.
        output_dir (str): Directory to save the output audio files.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint file not found at {ckpt_path}")
    
    config_path = os.path.join(os.path.dirname(ckpt_path), 'config.json')
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    
    model = TTS(language=language, config_path=config_path, ckpt_path=ckpt_path)
    
    for spk_name, spk_id in model.hps.data.spk2id.items():
        save_path = os.path.join(output_dir, spk_name, 'output.wav')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        model.tts_to_file(text, spk_id, save_path)
        print(f"Saved output for speaker {spk_name} at {save_path}")

if __name__ == "__main__":
    main()
