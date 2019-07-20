import click


from model.utils.data_generator import DataGenerator
from model.img2seq import Img2SeqModel
from model.utils.general import Config
from model.utils.text import Vocab
from model.utils.image import greyscale

from model.utils.text import load_formulas
from model.evaluation.text import score_files


@click.command()
@click.option('--results', default="results/small/", help='Dir to results')
def main(results):
    # restore config and model
    dir_output = results

    config_data = Config(dir_output + "data.json")
    config_vocab = Config(dir_output + "vocab.json")
    config_model = Config(dir_output + "model.json")

    vocab = Vocab(config_vocab)
    model = Img2SeqModel(config_model, dir_output, vocab)
    model.build_pred()
    # model.restore_session(dir_output + "model_weights/")

    # load dataset
    test_set = DataGenerator(path_formulas=config_data.path_formulas_test,
                             dir_images=config_data.dir_images_test,
                             img_prepro=greyscale,
                             max_iter=config_data.max_iter,
                             bucket=config_data.bucket_test,
                             path_matching=config_data.path_matching_test,
                             max_len=config_data.max_length_formula,
                             form_prepro=vocab.form_prepro,)

    # use model to write predictions in files
    config_eval = Config({
        "dir_answers": dir_output + "formulas_test/",
        "batch_size": 20
    })
    files, perplexity = model.write_prediction(config_eval, test_set)
    scores = score_files(files[0], files[1])
    scores["perplexity"] = perplexity

    msg = " || ".join(["{} is {:04.2f}".format(k, v) for k, v in scores.items()])
    model.logger.info("- Test Txt: {}".format(msg))


if __name__ == "__main__":
    main()
