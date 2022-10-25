import epitran

# "eng-Latn" processed separately
for epi_lang in ["aze-Latn", "ben-Beng", "ceb-Latn", "deu-Latn", 
    "hin-Deva", "kaz-Cyrl", "kir-Cyrl", "mlt-Latn", 
    "nld-Latn", "nya-Latn", "orm-Latn", "sna-Latn", 
    "swa-Latn", "swe-Latn", "tel-Telu", "tgk-Cyrl", 
    "tgl-Latn", "tuk-Latn", "uzb-Latn", "zul-Latn"]:

    lang = epi_lang[:3]
    print(lang)
    epi = epitran.Epitran(epi_lang)

    for split in ["form", "lemma"]:
        for part in ["trn", "tst", "dev"]:
            with open(f"/projects/tir4/users/mryskina/morphological-inflection/task0-data/split-by-{split}/grapheme/{lang}.{part}") as f:
                lines = f.readlines()
            with open(f"/projects/tir4/users/mryskina/morphological-inflection/task0-data/split-by-{split}/phoneme/{lang}.{part}", "w+") as f:
                for line in lines:
                    lemma, form, tags = line.split("\t")
                    f.write(f"{epi.transliterate(lemma)}\t{epi.transliterate(form)}\t{tags}")
