class Lista:
    class bertBaseMultilingual:
        nomeArtigo='mBERT'
        labels=[
            "Subjective",
            "Objective",
            "Assessment",
            "Plan"
            ]
        num_labels = 4
        epochs=4
        modelBERT = "bert-base-multilingual-cased"
        file_labels = "./rotulos.csv"
        random_state = 2018
        seed_val = 42
        out_dir = "bert-base-multilingual-cased_out/"

    class biobert:
        nomeArtigo='BioBERT'  
        labels=[
            "Subjective",
            "Objective",
            "Assessment",
            "Plan"
            ]
        num_labels = 4
        epochs=4    
        modelBERT = "dmis-lab/biobert-base-cased-v1.2"
        file_labels = "./rotulos.csv"
        random_state = 2018
        seed_val = 42
        out_dir = "biobert-base-cased-v1.2_out/"


    class distilbert:
        nomeArtigo='DistilBERT'
        labels=[
            "Subjective",
            "Objective",
            "Assessment",
            "Plan"
            ]
        num_labels = 4
        epochs=4
        modelBERT = "Geotrend/distilbert-base-pt-cased"
        file_labels = "./rotulos.csv"
        random_state = 2018
        seed_val = 42
        out_dir = "distilbert-base-pt-cased_out/"


    class bioBertPT:
        nomeArtigo='BioBERTpt'  
        labels=[
            "Subjective",
            "Objective",
            "Assessment",
            "Plan"
            ]
        num_labels = 4
        epochs=4
        modelBERT = "pucpr/biobertpt-all"
        file_labels = "./rotulos.csv"
        random_state = 2018
        seed_val = 42
        out_dir = "bioBertPT_out/"

    class bioBertPTRetreino:
        nomeArtigo='BioBERTptRT'  
        labels=[
            "Subjective",
            "Objective",
            "Assessment",
            "Plan"
            ]
        num_labels = 4
        epochs=4
        modelBERT = "./biobertRetreinada"
        file_labels = "./rotulos.csv"
        random_state = 2018
        seed_val = 42
        out_dir = "bioBERTptRetrain_out/"

