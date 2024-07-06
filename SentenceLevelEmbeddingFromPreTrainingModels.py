 # -*- coding: utf-8 -*-
import logging
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F

logging.basicConfig(filename='sentence_level_embeddings.log', level=logging.INFO, filemode='a', format='- %(message)s',
                    encoding='utf-8-sig')


def compare_models_tokenization_embeddings(model_names, sentence1, sentence2):
    logging.info(f"Input Sentence 1: {sentence1}")
    logging.info(f"Input Sentence 2: {sentence2}")
    print(f"Input Sentence 1: {sentence1}")
    print(f"Input Sentence 2: {sentence2}")

    results = {}

    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

        inputs1 = tokenizer(sentence1, return_tensors="pt", add_special_tokens=True, padding=True, truncation=True)
        inputs2 = tokenizer(sentence2, return_tensors="pt", add_special_tokens=True, padding=True, truncation=True)

        # Generate embeddings
        with torch.no_grad():
            outputs1 = model(**inputs1)
            outputs2 = model(**inputs2)

        # Extract the [CLS] token embeddings
        cls_embedding1 = outputs1.last_hidden_state[:, 0, :]
        cls_embedding2 = outputs2.last_hidden_state[:, 0, :]

        # Convert embeddings to list for logging
        emb_list1 = cls_embedding1.squeeze().tolist()
        emb_list2 = cls_embedding2.squeeze().tolist()

        # Calculate cosine similarity
        cosine_sim = F.cosine_similarity(cls_embedding1, cls_embedding2, dim=1).item()

        logging.info(f"Model: {model_name}")
        print(f"Model: {model_name}")
        logging.info(f"Cosine Similarity: {cosine_sim}")
        print(f"Cosine Similarity: {cosine_sim}")
        logging.info(f"CLS Embedding Sentence 1: {', '.join(f'{x:.4f}' for x in emb_list1)}")
        print(f"CLS Embedding Sentence 1: {', '.join(f'{x:.4f}' for x in emb_list1)}")
        logging.info(f"CLS Embedding Sentence 2: {', '.join(f'{x:.4f}' for x in emb_list2)}")
        print(f"CLS Embedding Sentence 2: {', '.join(f'{x:.4f}' for x in emb_list2)}")
        logging.info("-" * 30)
        print("-" * 30)

        results[model_name] = {
            "cosine_similarity": cosine_sim,
            "embeddings_shape1": cls_embedding1.shape,
            "embeddings_shape2": cls_embedding2.shape
        }

    logging.info("#" * 60)
    print("#" * 60)

    return results

model_names = [
    "bert-base-multilingual-cased",
    "xlm-roberta-base",
    "aubmindlab/bert-base-arabertv2",
    "aubmindlab/araelectra-base-discriminator"
]

# Sentences
sentence1 = """ أن الأم هي الشخص الذي يملك مكانة لا يمكن وصفها بالكلمات في حياة كل انسان. فهي تعتبر رمزا للحنان والعطاء، وتجسد الحب والرعاية بشكل فريد وخاص. أنها المخلوق الذي ينقل الحياه، وتعطيها المعنى الحقيقي والجميل.

تبدأ قصة الأم منذ اللحظة التي تشعر فيها بوجود الحياة داخلها. فتحمل بصبر وقوة هذه الأم المسؤولية العظيمة لحماية الجنين والعناية به خلال فترة الحمل التي تمتد لعدة اشهر. تعيش الأم في هذه الفترة تجربة فريدة من نوعها، تعبق بالتضحية والانتظار والامل.

عندما يأتي اليوم المنتظر لولادة الطفل، تجري الأم عملية ولادة شاقه، تختبر قوتها الجسدية والعاطفيه. فتتحمل الآلم والتعب بصبر لا يضاهي، وتضع كل جهودها لضمان وصول الطفل إلى الحياة بامان. وبمجرد ولادته، يمتلئ العالم بالفرح والحب الذي ينبعث من قلب الأم لهذا الكائن الصغير الذي تحملت كل هذه الصعاب من اجله.

منذ اللحظة الأولى لولادة الطفل، تتحول الأم إلى مربية ومعلمة وصديقه. فهي تهتم بكل تفاصيل حياة الطفل، تغسله وتغذيه وتهديه الحنان الذي يشعره بالأمان والراحه. تقضي الأم ساعات طويلة مع طفلها، تتفانى في تلبية احتياجاته ومساعدته على النمو والتطور. ولكن دور الأم لا يقتصر على السنوات الأولى من حياة الطفل """



sentence2 = """ أَنَّ الأُمَّ هِيَ الشَخْصُ الَّذِي يَمْلِك مَكانَةً لا يُمْكِن وَصْفُها بِالكَلِماتِ فِي حَياةِ كُلِّ انسان. فَهِيَ تُعْتَبَر رَمْزاً لِلحَنانِ وَالعطاء، وَتُجَسِّد الحُبَّ وَالرِعايَةَ بِشَكْلٍ فَرِيدٍ وَخاص. أَنَّها المَخْلُوقُ الَّذِي يَنْقُل الحياه، وَتُعْطِيها المَعْنَى الحَقِيقِيَّ وَالجميل.

تَبْدَأ قِصَّةُ الأُمِّ مُنْذُ اللَحْظَةِ الَّتِي تَشْعُر فِيها بِوُجُودِ الحَياةِ داخلها. فَتَحْمِل بِصَبْرٍ وَقُوَّةِ هٰذِهِ الأُمِّ المَسْؤُولِيَّةَ العَظِيمَةَ لِحِمايَةِ الجَنِينِ وَالعِنايَةِ بِهِ خلال فَتْرَةِ الحَمْلِ الَّتِي تَمْتَدّ لِعُدَّةِ اشهر. تَعِيش الأُمُّ فِي هٰذِهِ الفَتْرَةِ تَجْرِبَةً فَرِيدَةً مِن نوعها، تَعْبَق بِالتَضْحِيَةِ وَالاِنْتِظارِ وَالامل.

عَنْدَماً يَأْتِي اليَوْمُ المُنْتَظَرُ لِوِلادَةِ الطفل، تُجْرِي الأُمُّ عَمَلِيَّةَ وِلادَةٍ شاقه، تَخْتَبِر قُوَّتَها الجَسَدِيَّةَ وَالعاطفيه. فَتَتَحَمَّل الآلَمَ وَالتَعْبَ بِصَبْرٍ لا يضاهي، وَتَضَع كُلَّ جُهُودِها لِضَمانِ وُصُولِ الطِفْلِ إِلَى الحَياةِ بِامان. وَبِمُجَرَّدِ ولادته، يَمْتَلِئ العالَمُ بِالفَرَحِ وَالحُبِّ الَّذِي يَنْبَعِث مِن قَلْبِ الأُمِّ لِهٰذا الكائِنِ الصَغِيرِ الَّذِي تَحَمَّلَت كُلَّ هٰذِهِ الصِعابِ مِن اجله.

مُنْذُ اللَحْظَةِ الأَوْلَى لِوِلادَةِ الطفل، تَتَحَوَّل الأُمُّ إِلَى مُرَبِّيَةٍ وَمُعَلِّمَةٍ وَصديقه. فَهِيَ تَهْتَمّ بِكُلِّ تَفاصِيلِ حَياةِ الطفل، تَغْسِله وَتُغَذِّيه وَتُهْدِيه الحَنانَ الَّذِي يَشْعُره بِالأَمانِ وَالراحه. تَقْضِي الأُمُّ ساعاتٍ طَوِيلَةً مع طفلها، تَتَفانَى فِي تَلْبِيَةِ اِحْتِياجاتِهِ وَمُساعَدَتِهِ عَلَى النُمُوِّ وَالتطور. وَلٰكِن دَوْرَ الأُمِّ لا يَقْتَصِر عَلَى السَنَواتِ الأَوْلَى مِن حَياةِ الطِفْلِ"""



# # Sentences
# sentence1 = """ أن الأم هي الشخص الذي يملك مكانة لا يمكن وصفها بالكلمات في حياة كل انسان. فهي تعتبر رمزا للحنان والعطاء. """
#
#
#
# sentence2 = """ أَنَّ الأُمَّ هِيَ الشَخْصُ الَّذِي يَمْلِك مَكانَةً لا يُمْكِن وَصْفُها بِالكَلِماتِ فِي حَياةِ كُلِّ انسان. فَهِيَ تُعْتَبَر رَمْزاً لِلحَنانِ وَالعطاء."""


results = compare_models_tokenization_embeddings(model_names, sentence1, sentence2)
print("Results have been logged to 'sentence_level_embeddings.log'")

