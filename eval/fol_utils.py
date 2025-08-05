from eval.tasks.utils import convert_to_nltk_rep


def reformat_fol_samples_train(train_dataset):
    def reformat_fol_sample(sample):
        sample["premises-FOL"] = [
            convert_to_nltk_rep(premise) for premise in sample["premises-FOL"]
        ]
        return sample

    return train_dataset.map(reformat_fol_sample)

def add_conclusion_fols_train(train_dataset):
    train_conclusion_fols = {
            23: "HigherRank(RealMadrid, Barcelona)",
            60: "-OlympicGoldMedalWinner(Amy) -> NobelLaureate(Amy)",
            125: "-Dispensable(Worksheet)",
            148: "FolkSong(Inception)",
            261: "MakeGoodBreakfast(Luke)",
            263: "exists x. (Develops(Ets, x) & For(x, k-OneTwoandhighereducation)) & exists x. (Develops(Ets, x) & AssociatedWith(x, Entrytouseducationinstitutions))",
            275: "ContributeToCountry(James)",
            299: "GetRhythmRight(John)",
            683: "exists x. (BRICS(x) & Speaks(x, Hindi))",
            684: "Film(Hamilton)",
            850: "-Liked(Leo, Charlie) & -Cares(Charlie, Leo)",
            853: "Won(Threebodyproblem, Hugoaward)",
            886: "Dagfinn(DagfinnAarskog)",
            892: "PartOf(Minsk, Scottishpremiership)",
            930: "-Locate(Boves, Europe)",
            980: "(InvitedTakePhoto(James) & -HappyCommunicate(James)) | (-InvitedTakePhoto(James) & HappyCommunicate(James))",
        }
    conclusions = [None for _ in range(len(train_dataset))]
    for index, conclusion_fol in train_conclusion_fols.items():
        conclusions[index] = conclusion_fol
    train_dataset = train_dataset.add_column("conclusion-FOL", conclusions)
    return train_dataset

def add_cot_train(train_dataset):
    train_cots = {
            23: "Let's think step by step. We want to evaluate if in La Liga 2021-2022, Real Madrid ranks higher than Barcelona. From premise 1, we know that a La Liga soccer team ranks higher than another if it receives more points. From premise 4, we know that in La Liga 2021-2022, Real Madrid received more points than Barcelona. Therefore, in La Liga 2021-2022, Real Madrid received more points than Barcelona, so Real Madrid ranks higher than Barcelona, so the statement is true.\nANSWER:\tTrue",
            60: "Let's think step by step. We want to evaluate the statement \"if Amy is not an Olympic gold medal winner, then Amy is a Nobel laureate\". Let's assume that Amy is not an Olympic gold medal winner. This doesn't tell us anything about whether Amy is a Nobel laureate, so the statement isn't true, meaning it is either False or Uncertain. To distinguish between the two, notice that we could have a scenario where Amy is neither an Olympic gold medal winner nor a Nobel laureate. None of the premises are violated in this case. This means the statement must be false.\nANSWER:\tFalse",
            125: "Let's think step by step. We want to evaluate if a worksheet is not dispensable. From premise 6, we know that a worksheet is either paper or is environment-friendly. If it is paper, then from premise 3, a worksheet is woodware, and from premise 2, a worksheet is dispensable. If it is environment-friendly, we know it is good from premise 5, but we know nothing about whether it is dispensable. Therefore, we don't know if a worksheet is dispensible or not, so the statement is uncertain.\nANSWER:\tUncertain",
            148: "Let's think step by step. We want to evaluate if Inception is a folk song. We know that Inception is a sci-fi movie. Since all movies are videos and Inception is a movie, it is a video, which means it is visual. On the other hand, we know that all folk songs are songs, and no songs are visual, so no folk songs are visual. Therefore, since Inception is visual but no folk songs are visual, we know that Inception cannot be a folk song, so the statement is false.\nANSWER:\tFalse",
            261: "Let's think step by step. We want to evaluate if Luke can make a good breakfast. From the last premise, we know that Luke can make cookies, scrambled eggs, and muffins. Since Luke can make cookies and muffins, they are a baker. Now, combining the information we have, since Luke is a baker and can make scrambled eggs, this means that they can make a good breakfast. Therefore, Luke can make a good breakfast, so the statement is true.\nANSWER:\tTrue",
            263: "Let's think step by step. We want to evaluate if ETS develops assessments for K-12 statewide as well as entry to US tertiary and quaternary educatiand doon institutions. We know that ETS develops assessments for K-12 statewide. We also know that ETS develops assessments associated with entry to the US tertiary and quaternary education institutes. Therefore, both parts of the conclusion are true, and the statement is true.\nANSWER:\tTrue",
            275: "Let's think step by step. We want to evaluate if James contributes to the country. Let's think about what we know about James. First, we know that James was either sentenced for thief or stayed in prison. However, this doesn't tell us anything about whether James contributed to the country. Second, we know that James either had a bad record in the local state or that he was respected by others. However, the premises don't tell us anything about the relationship between having a bad record and contributing to the country. Therefore, it is uncertain whether James contributes to the country.\nANSWER:\tUncertain",
            299: "Let's think step by step. We want to evaluate if John can get the rhythms right. We know that John is a student learning piano. Since all students learning piano can strike the right notes, John can strike the right notes. Since all students who can strike the right notes can get the rhythms right and John can strike the right notes, John can get the rhythms right, so the conclusion is true.\nANSWER:\tTrue",
            683: "Let's think step by step. We want to evaluate if there is a person from BRICS speaking Hindi. We know that there is an Indian, and since India is one of BRICS, we know that there is an Indian in BRICS. Furthermore, we know that they speak either Hindi or English, however, we don't know which one. Therefore, there could be a person in BRICS speaking Hindi, or there could not. Therefore, it is uncertain whether there is a person from BRICS speaking Hindi.\nANSWER:\tUncertain",
            684: "Let's think step by step. We want to evaluate if Hamilton is a film. Since Daveed Diggs played two roles in the musical Hamilton, Hamilton is a musical. Since musicals are not films and Hamilton is a musical, Hamilton is not a film, and the conclusion is false.\nANSWER:\tFalse",
            850: "Let's think step by step. We want to evaluate if Charlie does not like Leo and does not care for Leo. Let's first evaluate if Charlie does not like Leo. We know Charlie has a naughty pet named Leo. Since pets who are naughty are not liked as much, Charlie does not like Leo. Now, let's evaluate if Charlie cares for Leo. We know that if a person has a pet, they care for that pet. Since Leo is Charlie's pet, Charlie cares for Leo. Therefore, Charlie does not like Leo but cares for Leo, so the second part of the conclusion is false, which means the entire conclusion is false.\nANSWER:\tFalse",
            853: "Let's think step by step. We want to evaluate if the Three Body Problem won the Hugo Award. The only thing we know about the Hugo Award is that some books that have won the Hugo Award were written by Cixin Liu. However, we know nothing about whether The Three Body Problem was written by Cixin Liu, so the conclusion is uncertain.\nANSWER:\tUncertain",
            886: "Let's think step by step. We want to evaluate if Dagfinn is Dagfinn Aarskog's given name. We know that Dagfinn is a given name, and that notable people with the given name Dagfinn includes Dagfinn Aarskog, which means that Dagfinn is Dagfinn Aarskog's given name, so the conclusion is true.\nANSWER:\tTrue",
            892: "Let's think step by step. We want to evaluate if Minsk joined the Scottish Premiership. We know that Minsk and St Johnstone are different teams and that St Johnstone is part of the Scottish Premiership, but we don't know anything about whether or not Minsk joined the Scottish Premiership from the premises. Therefore, the conclusion is uncertain.\nANSWER:\tUncertain",
            930: "Let's think step by step. We want to evaluate if Boves is not in Europe. We know that Boves is a railway station located in France. We also know that since France is a European country, France is located in Europe. Furthermore, we know that if A is located in B and B is located in C, then A is located in C. Therefore, we know that because Boves is located in France and France is located in Europe, that means Boves is located in Europe. Therefore, the conclusion is false.\nANSWER:\tFalse",
            980: "Let's think step by step. We want to evaluate if James is either invited to take a photo with the audience or happy to communicate with each other during the dinner. We know that James does not attend the conference in person and is not provided with souvenirs. There are no premises that apply to people who do not attend the conference. Since James is not provided with souvenirs, since all who attended the conference in person are provided with souvenirs, we know that James did not attend the conference in person. However, we don't know anything else, so it is possible that James was neither invited to take a photo with the audience nor happy to communicate during the dinner. Therefore, the conclusion is false.\nANSWER:\tFalse",
        }
    cots = [None for _ in range(len(train_dataset))]
    for index, cot in train_cots.items():
        cots[index] = cot
    train_dataset = train_dataset.add_column("cot", cots)
    return train_dataset