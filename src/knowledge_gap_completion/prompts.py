def assemble_prompt_knowledge_question_generation():

    prefix_date = """You are a helpful assistant. You are given context information about an image. The date of the image is unknown. Your task is to generate one to three world knowledge questions that will help determine when the image was taken. The questions should either start with “When” or “On which date”. The question should be self-contained and specific enough to be answerable based on world knowledge. If no self-contained and specific question can be generated, your response should be “No questions can be generated given the context”. You are provided with examples below."""

    demo_prompts_date = [
        """Context information:
        Things: A neon horse. 
        Event: The Chinese new year celebration for the Year of the Horse.
        Location: Fo Guang Shan Dong Zen temple in Klang, Malaysia.
        Motivation: To report on the celebrations for the start of the Year of the Horse in Malaysia.

        Generated questions (up to 3): """,

        """Context information:
        People: Billy Sharp of Reading and a team mate, and 2 players of Nottingham Forest.
        Things: A football field.
        Event: A football game between Reading and Nottingham Forest, likely during the Championship 2011-2012 season.
        Motivation: To report on an action involving Billy Sharp and other players during a game between Reading and Nottingham Forest.

        Generated questions (up to 3): """,

        """Context information:
        People: An unknown woman.
        Things: A Playstation VR headset and move controllers.
        Event: The woman is playing a Playstation VR game with a headset and move controllers.
        Motivation: To report on the Playstation VR.

        Generated questions (up to 3): """,

        """Context information:
        People: A cyclist.
        Things: The Black Cultural Archives building.
        Location: Brixton, London.

        Generated questions (up to 3): """,
    ]

    demo_answers_date = [
        """Q1: When was the most recent Year of the Horse?
        Q2: On which date was the start of the most recent Year of the Horse celebrated?
        Q3: On which date was the most recent Year of the Horse celebration in Klang, Malaysia?""",

        """Q1: On which date did the game between Reading and Notthingham Forest take place during the Championship 2011-2012 season?""",

        """Q1: When was the Playstation VR first released?""",

        """Q1: When did the Black Cultural Archives building open in Brixton?""",
    ]


    prefix_location = """You are a helpful assistant. You are given context information about an image. The location of the image is unknown. Your task is to generate one to three world knowledge questions that will help determine where the image was taken. The question should start with “Where”, “In which country”, or “In which place”. The question should be self-contained and specific enough to be answerable based on world knowledge. If no self-contained and specific question can be generated, your response should be “No questions can be generated given the context”. You are provided with examples below."""

    demo_prompts_location = [
        """Context information:
        Things: A neon horse. 
        Event: The Chinese new year celebration for the Year of the Horse.
        Motivation: To report on the celebrations for the start of the Year of the Horse in Malaysia.

        Generated question: """,

        """Context information:
        People: Billy Sharp of Reading and a team mate, and 2 players of Nottingham Forest.
        Things: A football field.
        Event: A football game between Reading and Nottingham Forest, likely during the Championship 2011-2012 season.
        Motivation: To report on an action involving Billy Sharp and other players during a game between Reading and Nottingham Forest.

        Generated question: """,

        """Context information:
        Things: Trees, a street. A bright light in the background, which could indicate a fire.
        Event: A home was burned down at Satellite Beach during Hurricane Matthew.
        Date: Hurricane Matthew happened in October 2016.
        Motivation: To report on the damages caused by Hurricane Matthew.

        Generated question: """,

        """Context information:
        People: Novak Djokovic.
        Things: A tennis court, a tennis racket.
        Event: Novak Djokovic jumping on the court during a game with Rafael Nadal at the US Open 2013.
        Date: September 9, 2013.
        Motivation: To report on the final of the US Open 2013 opposing Novak Djokovic to Rafael Nadal.

        Generated question: """,
    ]

    demo_answers_location = [
        """No questions can be generated given the context information.""",

        """Q1: Where did the game between Reading and Nottingham Forest take place during the Championship 2011-2012 season?
        Q2: In which place does Reading play its home games?
        Q3: In which place does Nottingham Forest play its home games?""",

        """Q1: Where is Satellite Beach located?""",

        """Q1: Where did the US Open 2013 final take place?""",
    ]


    messages_date = [{"role": "system", "content": prefix_date}]

    for i in range(len(demo_prompts_date)):
        messages_date.append({"role": "user", "content": demo_prompts_date[i]})
        messages_date.append({"role": "assistant", "content": demo_answers_date[i]})
        
    messages_location = [{"role": "system", "content": prefix_location}]

    for i in range(len(demo_prompts_location)):
        messages_location.append({"role": "user", "content": demo_prompts_location[i]})
        messages_location.append({"role": "assistant", "content": demo_answers_location[i]})
    
    return messages_date, messages_location



def assemble_prompt_knowledge_qa():
    prefix_date = """You are a helpful assistant. You are given a question that requires world knowledge to be answered. Your task is to provide a specific answer to the question in 1 or 2 sentences based on available knowledge from Wikipedia. Your answer should be a date at the day, month or year level. If the question cannot be answered based on the available knowledge, your response should be “Unknown”. You are provided with examples below."""

    demo_prompts_date = [
        """Wikipedia knowledge: The year of the Wood Horse started on 31 January 2014 and ended on 18 February 2015.
        Question: When was the most recent Year of the Horse? 

        Answer: """,

        """Wikipedia knowledge: The BCA's new building in Brixton, opened in 2014, enables access to the archive collection, provides dedicated learning spaces and mounts a programme of exhibitions and events.
        Question: When did the Black Cultural Archives building open in Brixton?

        Answer: """,

        """Wikipedia knowledge: Reading confronted Nottingham Forest on two occasions for the 2011-2012 Championship, on the 1st of November 2011 and on the 17th of April 2012.
        Question: On which date did the game between Reading and Nottingham Forest take place during the Championship 2011-2012 season?

        Answer: """,

        """Wikipedia knowledge: On October 13, 2016, Sony released the PlayStation VR with the price of $399 in the US, €399 in Europe, £349 in the UK, and ¥44,980 in Japan. On April 16, 2019, Mark Cerny confirmed that the PlayStation VR would be compatible with the PlayStation 5.
        Question: When was the Playstation VR first released?

        Answer: """,
    ]

    demo_answers_date = [
        """The most recent Year of the Horse started on 31 January 2014 and ended on 18 February 2015.""",

        """The building of the Black Cultural Archives in Brixton opened in 2014.""",

        """There were 2 games between Reading and Nottingham Forest in 2011-2012 Championship season. One on 1 November 2011, and the other on 17 April 2012.""",

        """The Playstation VR was released on 13 October 2016.""",
]
    
    prefix_location = """You are a helpful assistant. You are given a question that requires world knowledge to be answered. Your task is to provide a specific answer to the question in 1 or 2 sentences based on available knowledge from Wikipedia. If the question cannot be answered based on the available knowledge, your response should be “Unknown”. You are provided with examples below."""

    demo_prompts_location = [
        """Wikipedia knowledge: Reading confronted Nottingham Forest on two occasions for the 2011-2012 Championship, once in Madejski Stadium in Reading, Berkshire, UK, and the other at City Ground in Westbrigdford, Nottingham shire, UK.
        Question: Where did the game between Reading and Nottingham Forest take place during the Championship 2011-2012 season?

        Answer: """,

        """Wikipedia knowledge: Dong Zen Temple is located in Jenjarom within the Kuala Langat district of Selangor state
        Question: Where is Fo Guang Shan Dong Zen temple located?

        Answer: """,

        """Wikipedia knowledge: It took place at the USTA Billie Jean King National Tennis Center, and ran from August 26 to September 9.
        Question: Where did the US Open 2013 final take place?

        Answer: """,

        """Wikipedia knowledge: Satellite Beach is a coastal city in Brevard County, Florida, U.S.
        Question: Where is Satellite Beach located?

        Answer: """,
    ]

    demo_answers_location = [
        """There were two games between Reading and Nottingham Forest that took place during the Championship 2011-2012 season. One was at Madejski Stadium in Reading, Berkshire, UK, and the other at the City Ground in Westbrigdford, Nottingham shire, UK.""",

        """The Fo Guang Shan Dong Zen temple is located in Jenjarom, Selangor, Malaysia.""",

        """The US Open 2013 final took place at USTA Billie Jean King National Tennis Center, New York, United States.""",

        """Satellite Beach is a coastal city located in Brevard County, Florida, United States.""",
    ]


    messages_date = [{"role": "system", "content": prefix_date}]

    for i in range(len(demo_prompts_date)):
        messages_date.append({"role": "user", "content": demo_prompts_date[i]})
        messages_date.append({"role": "assistant", "content": demo_answers_date[i]})


    messages_location = [{"role": "system", "content": prefix_location}]

    for i in range(len(demo_prompts_location)):
        messages_location.append({"role": "user", "content": demo_prompts_location[i]})
        messages_location.append({"role": "assistant", "content": demo_answers_location[i]})
    
    return messages_date, messages_location


def assemble_prompt_knowledge_validation():

    prefix_date = """You are a helpful assistant. You are given context information about an image, as well as world knowledge information. Your task is to estimate when the image was taken or provide a plausible time range using the context and world knowledge information. If the date cannot be derived from the context and world knowledge, your response should be “Unknown”. You are provided with examples below."""

    demo_prompts_date = [
        """Context: 
        Things: A neon horse. 
        Event: The Chinese new year celebration for the Year of the Horse.
        Location: Fo Guang Shan Dong Zen temple in Klang, Malaysia.
        Motivation: To report on the celebrations for the start of the Year of the Horse in Malaysia.

        World knowledge:
        The most recent Year of the Horse started in 31 January 2014 and ended in 18 February 2015. The start of the Year of the Horse was celebrated on 31 January 2014 in Malaysia.

        When was the image taken?

        Answer: """,

        """Context: 
        People: Billy Sharp of Reading and a teammate, and 2 players of Nottingham Forest.
        Things: A football field.
        Event: A football game between Reading and Nottingham Forest, likely during the Championship 2011-2012 season.
        Motivation: To report on an action involving Billy Sharp and other players during a game between Reading and Nottingham Forest.

        World knowledge:   
        There were 2 games between Reading and Notthingham Forest in 2011-2012 Championship season. One on 1 November 2011, the other on 17 April 2012.

        When was the image taken?

        Answer: """,

        """Context: 
        People: An unknown woman.
        Things: A Playstation VR headset and move controllers.
        Event: The woman is playing a Playstation VR game with a headset and move controllers.
        Motivation: To report on the Playstation VR.

        World knowledge:   
        The Playstation VR was released on 13 October 2016.

        When was the image taken?

        Answer: """,

        """Context: 
        People: A cyclist.
        Things: The Black Cultural Archives building.
        Location: Brixton, London.

        World knowledge:   
        The building of the Black Cultural Archives in Brixton opened officially on 24 July 2014.

        When was the image taken?

        Answer: """,
    ]

    demo_answers_date = [
        """31 January 2014.""",

        """Either on 1 November 2011 or 17 April 2022.""",

        """13 October 2016 or after.""",

        """24 July 2014 or after.""",
    ]

    prefix_location = """You are a helpful assistant. You are given context information about an image, as well as world knowledge information. Your task is to estimate where the image was taken using the context and world knowledge information. If the location cannot be derived from the context and world knowledge, your response should be “Unknown”. You are provided with examples below."""

    demo_prompts_location = [
        """Context: 
        Things: A neon horse. 
        Event: The Chinese new year celebration for the Year of the Horse at Fo Guang Shan Dong Zen temple.
        Motivation: To report on the celebrations for the start of the Year of the Horse at Fo Guang Shan Dong Zen temple.

        World knowledge:   
        The Fo Guang Shan Dong Zen temple is located in Jenjarom, Selangor, Malaysia.

        Where was the image taken?

        Answer: """,

        """Context: 
        People: Billy Sharp of Reading and a teammate, and 2 players of Nottingham Forest.
        Things: A football field.
        Event: A football game between Reading and Nottingham Forest, likely during the Championship 2011-2012 season.
        Motivation: To report on an action involving Billy Sharp and other players during a game between Reading and Nottingham Forest.

        World knowledge:   
        There were two games between Reading and Nottingham Forest that took place during the Championship 2011-2012 season. One was at Madejski Stadium in Reading, Berkshire, UK, and the other at the City Ground in Westbrigdford, Nottingham shire, UK.

        Where was the image taken?

        Answer: """,

        """Context: 
        Things: Trees, a street. A bright light in the background, which could indicate a fire.
        Event: A home was burned down at Satellite Beach during Hurricane Matthew.
        Date: Hurricane Matthew happened in October 2016.
        Motivation: To report on the damages caused by Hurricane Matthew.

        World knowledge:   
        Satellite Beach is a coastal city located in Brevard County, Florida, United States.

        Where was the image taken?

        Answer: """,

        """Context: 
        People: Novak Djokovic.
        Things: A tennis court, a tennis racket.
        Event: Novak Djokovic jumping on the court during a game with Rafael Nadal at the US Open 2013.
        Date: September 9, 2013.
        Motivation: To report on the final of the US Open 2013 opposing Novak Djokovic to Rafael Nadal.

        World knowledge:   
        The US Open 2013 final took place at USTA Billie Jean King National Tennis Center, New York, United States.

        Where was the image taken?

        Answer: """,
    ]

    demo_answers_location = [
        """At the Fo Guang Shan Dong Zen temple in Jenjarom, Selangor, Malaysia.""",

        """Either in Madejski stadium or City Ground.""",

        """Satellite Beach, Brevard County, Florida, United States.""",

        """USTA Billie Jean King National Tennis Center, New York, United States.""",
    ]

    messages_date = [{"role": "system", "content": prefix_date}]

    for i in range(len(demo_prompts_date)):
        messages_date.append({"role": "user", "content": demo_prompts_date[i]})
        messages_date.append({"role": "assistant", "content": demo_answers_date[i]})

    messages_location = [{"role": "system", "content": prefix_location}]

    for i in range(len(demo_prompts_location)):
        messages_location.append({"role": "user", "content": demo_prompts_location[i]})
        messages_location.append({"role": "assistant", "content": demo_answers_location[i]})

    return messages_date, messages_location