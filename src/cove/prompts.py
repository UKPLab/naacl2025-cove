def generate_QA_demos_with_captions():
    prefix_all = """You are a helpful assistant that answers questions about an image based on captions from the web, automatically generated captions, and visual entities.
    You are given captions from web pages that used the image.
    You are also given two automatically generated captions. The first caption provides a global description of the image, sometimes including people, but without naming them. The second caption provides more details about the people in the image, including their names and short biographies when that information is available.
    You are also given visual entities which may be relevant to the image, without certitude.
    Here are a few examples:"""
    prefix_QA = """
    Web captions that may be relevant: {}
    Caption 1 - Global description: {}
    Caption 2 - More details about the people in the image: {}
    Visual entities that may be relevant, without certitude: {}
    Question: {} Answer in one sentence. Answer with one word ("unknown") if the information is not provided.
    Answer: """

    people_Q = "Who is shown in the image?"
    apo_Q = "Which animal, plant, building, or object are shown in the image?"
    event_Q = "Which event is depicted in the image?"
    date_Q = "When was the image taken?"
    location_Q = "Where was the image taken?"
    motivation_Q = "Why was the image taken?"
    source_Q = "Who is the source of the image?"

    questions = [people_Q, apo_Q, event_Q, date_Q, location_Q, motivation_Q, source_Q]

    demo1_captions = {
        'visual_entities': "Sik Sik Yuen Wong Tai Sin Temple, Kuta, 馬來西亞佛光山東禪寺 Fo Guang Shan Dong Zen Temple, Klang, Chinese New Year, Dragon dance, Horse, Performance art, Image, Incense, Parade, Hong Kong, Bali, Kuta",
        'caption_1': "The image shows a neon horse during the night.",
        'caption_2': "No people are shown in the image.",
        'web': "caption 1: Chinese new year - in pictures | Life and style | The Guardian, caption 2: A neon horse at the Fo Guang Shan Dong Zen temple in Klang, Malaysia welcomes in the Year of the Horse."
    }
    demo1_answers = [
        "Unknown.",
        "A neon horse.",
        "The Chinese new year celebration for the Year of the Horse.",
        "The image was taken for the celebrations of the start of the Year of the Horse. The last Year of the Horse started on January 31, 2014. Hence, the date is likely January 31, 2014.",
        "Fo Guang Shan Dong Zen temple in Klang, Malaysia.",
        "To report on the celebrations for the start of the Year of the Horse in Malaysia.",
        "The Guardian."
    ]

    demo2_captions = {
        'visual_entities': "United States, Politics, Governor of New Jersey, Governor, Conservatism, Commentator, Candidate, Speech, Chris Christie, Jeffrey Sachs",
        'caption_1': "The image shows a man who is a politician standing behind a lectern at a political rally event.",
        'caption_2': "The image shows Donald Trump. He was the 45th President of the United States.",
        'web': "caption 1: Donald Trump says he will spend $2m a week on TV campaign ads | Donald Trump | The Guardian, caption 2: Donald Trump told a campaign rally in Council Bluffs, Iowa: ‘I don’t want to take any chances.’, caption 3: Donald Trump speaks during a campaign rally in Council Bluffs, Iowa, caption 4: How Donald Trump’s Words Create Emergencies | The Nation, caption 5: Donald Trump Iowa Campaign Stop, caption 6: Donald Trump speaks to a crowd during a Republican presidential primary campaign rally in Council Bluffs, Iowa, on December 29, 2015.  , caption 7: USA: 'Nobody respects women more than Donald Trump' says Trump at rally - YouTube, caption 8: USA: 'Trump is disgusting' appears in sky over Pasadena's Rose Parade - YouTube"
    }
    demo2_answers = [
        "Donald Trump.",
        "Unknown.",
        "A campaign rally of Donald Trump for the Republican presidential primary of 2015.",
        "December 29, 2015.",
        "Council Bluffs, Iowa, USA.",
        "To report on a political campaign rally of Donald Trump in the context of the Republican presidential primary.",
        "Unknown."
    ]

    demo3_captions = {
        'visual_entities': "Black Cultural Archives, Windrush Square, Archive, October Calling, October Calling, Façade, Cycling, Bicycle, National Lottery Heritage Fund, Recreation, Funding, Money, Letter, London",
        'caption_1': "The image shows a man riding a bicycle on a sidewalk next to a building.",
        'caption_2': "The person in the image cannot be identified precisely.",
        'web': "caption 1: London's Black Cultural Archives get £200,000 stopgap funding for survival | London | The Guardian, caption 2: MPs urge government to secure future of Black Cultural Archives | London | The Guardian"
    }
    demo3_answers = [
        "A cyclist.",
        "The Black Cultural Archives building.",
        "Unknown.",
        "Unknown.",
        "London. More specifically, the Black Cultural Archives are located in Brixton, London, England.",
        "Unknown.",
        "The Guardian."
    ]

    demo4_captions = {
        'visual_entities': "Tangles Hair Studio and Color Lounge, Titusville, Merritt Island, Satellite Beach, Effects of Hurricane Matthew in Florida, Tangles and Waves Children's Salon, Florida Today, Landfall, Storm, Tropical Cyclone, Protect 13, Sunrise, Brevard County",
        'caption_1': "The image shows a person walking in front of a tree with a bright light in the background, possibly indicating a fire.",
        'caption_2': "The person in the image cannot be identified precisely.",
        'web': "caption 1: Hurricane Matthew: FLORIDA TODAY staff&#39;s experience, caption 2: Seen through snapped branches across the street, a Satellite Beach home burned down during Hurricane Matthew., caption 3: Fastly error: unknown domain wlna-webservice.gannettdigital.com"
    }
    demo4_answers = [
        "Unknown.",
        "Trees, a street, a bright light in the background, which could indicate a fire.",
        "A home was burned down at Satellite Beach during Hurricane Matthew.",
        "Hurricane Matthew happened in October 2016.",
        "Satellite Beach, which is located in Florida, USA.",
        "To report on the damages caused by Hurricane Matthew in Florida.",
        "Florida Today."
    ]

    demo5_captions = {
        'visual_entities': "Track Shoe, Activewear, Shoe, Personal protective equipment, Sports shoes, Team sport, Athlete, Player, Competition, Sneakers",
        'caption_1': "The image shows a tennis player jumping on a tennis field after hitting the ball. Spectators are sitting behind and the sponsor American Express appears on the banners.",
        'caption_2': "The image shows Novak Djokovic. He is a famous Serbian tennis player who won several tournaments.",
        'web': "caption 1: Images tagged &quot;novak-djokovic&quot; - | Tennis.it, caption 2: 02-rafael-nadal-novak-djokovic-us-open-tennis, caption 3: US OPEN 2013: RAFA NADAL DAL DIVANO AL TRIONFO - Tennis.it, caption 4: 02-rafael-nadal-novak-djokovic-us-open-tennis"
    }
    demo5_answers = [
        "Novak Djokovic.",
        "A tennis court, a tennis racket.",
        "Novak Djokovic jumping on the court during a game with Rafael Nadal at the US Open 2013.",
        "September 9, 2013.",
        "US Open at USTA Billie Jean King National Tennis Center in Flushing Meadows, Queens, New York City.",
        "To report on the final of the US Open 2013 opposing Novak Djokovic to Rafael Nadal.",
        "Unknown."
    ]

    demo6_captions = {
        'visual_entities': "Gender identity, Sexual orientation, Discrimination, Asylum seeker, Human sexuality, Social equality, Photojournalist",
        'caption_1': "The image shows a man in a bathrobe sitting on a couch smoking. There is a kitchen behind him and a table with a mirror next to him.",
        'caption_2': "The person in the image is not identified.",
        'web': "caption 1: Photojournalist Documents Lives of LGBTQ Asylum Seekers in Mideast, caption 2: Sourena, who left Iran and applied for resettlement in Canada via the UNHCR, sits in his kitchen in Isparta, western Turkey., caption 3: Image:, caption 4: Seeking home: The lives of gay and transgender asylum seekers of the Middle East - The Washington Post"
    }
    demo6_answers = [
        "Sourena, an LGBTQ asylum seeker in the Mideast.",
        "A kitchen, a table with a mirror, a cigarette, a bathrobe.",
        "Sourena is smoking in his kitchen.",
        "Unknown.",
        "Isparta, Western Turkey.",
        "To report on the life of Sourena, an LGBTQ asylum seeker who left Iran and applied for resettlement in Canada via the UNHCR.",
        "Washington Post."
    ]

    demo7_captions = {
        'visual_entities': "Team sport, International rules football, Competition, Player, Defensive tackle, Tournament, Sports venue, Team, Tackle, Product",
        'caption_1': "The image shows four players in action on a football field, two from each team. One team is playing in red while the other team is in blue and white horizontal stripes.",
        'caption_2': "One of the players is Billy Sharp from Reading. The other players are not directly identifiable based on the image content.",
        'web': "caption 1: Reading sign Garath McCleary from Nottingham Forest | Reading | The Guardian, caption 2: Marcus Tudgay gives Nottingham Forest the edge over Reading | Championship 2011-12 | The Guardian"
    }
    demo7_answers = [
        "Billy Sharp of Reading and a team mate, and 2 players of Nottingham Forest.",
        "A football field.",
        "A football game between Reading and Nottingham Forest, likely during the Championship 2011-2012 season.",
        "Championship 2011-2012.",
        "Unknown.",
        "To report on an action involving Billy Sharp and other players during a game between Reading and Nottingham Forest.",
        "The Guardian."
    ]

    demo8_captions = {
        'visual_entities': "HSBC Bank USA, HSBC Bank, Bank, HSBC, Credit card, Online banking, ABA routing transit number, Wire transfer, Meezan Bank, Financial services, HSBC Bank (Vietnam) limited, Loan, Money",
        'caption_1': " The image shows the entrance of a building with the HSBC logo on the front door.",
        'caption_2': "No people are shown in the image.",
        'web': "caption 1: <b>HSBC Bank</b> Customer Care Numbers Toll Free Phone Numbers, caption 2: HSBC-bank-logo-007, caption 3: Report: <b>HSBC</b> allowed money laundering that likely funded terror ..., caption 4: A HSBC bank logo is highlighted by the sun in London in this file photo taken March 1, 2010., caption 5: Meezan Bank set to acquire <b>HSBC Pakistan</b> - Newspaper - DAWN ..., caption 6: HSBC Bank Pakistan. -File photo, caption 7: HSBC pulls investment from Sinar Mas after Greenpeace protest | Guardian sustainable business | The Guardian, caption 8: Slippery shares: HSBC ditches its shares in palm oil after Greenpeace protest. Photograph: Reuters, caption 9: hsbc, caption 10: Whistleblower wins 13-year campaign against HSBC | Credit cards | The Guardian, caption 11: HSBC will set up a £4m compensation scheme for people who lost out financially due to excessive card charges., caption 12: Whistleblower wins out over HSBC charges; plus, Sky ditches the dishes | Money | The Guardian, caption 13: Thousands of customers will be compensated for excessive charges thanks to 59-year-old Nicholas Wilson., caption 14: Business week in pictures | Business | The Guardian, caption 15: Week in business: HSBC bank logo in London, caption 16: Hsbc Bank Logo High Resolution Stock Photography and Images - Alamy, caption 17: HSBC Allowed Narcos To Launder Billions ~ Borderland Beat, caption 18: HSBC-bank-logo-007 | Customer Care Help"
    }
    demo8_answers = [
        "Unknown.",
        "The entrance of a building with the HSBC logo.",
        "Unknown.",
        "March 1, 2010.",
        "London.",
        "Unknown.",
        "Unknown."
    ]

    prompts_demo1 = [[{"role": "user", "content": prefix_QA.format(demo1_captions['web'], demo1_captions['caption_1'], demo1_captions['caption_2'], demo1_captions['visual_entities'], questions[ix])}, {"role": "assistant", "content": demo1_answers[ix]}] for ix in range(len(questions))]
    prompts_demo2 = [[{"role": "user", "content": prefix_QA.format(demo2_captions['web'], demo2_captions['caption_1'], demo2_captions['caption_2'], demo2_captions['visual_entities'], questions[ix])}, {"role": "assistant", "content": demo2_answers[ix]}] for ix in range(len(questions))]
    prompts_demo3 = [[{"role": "user", "content": prefix_QA.format(demo3_captions['web'], demo3_captions['caption_1'], demo3_captions['caption_2'], demo3_captions['visual_entities'], questions[ix])}, {"role": "assistant", "content": demo3_answers[ix]}] for ix in range(len(questions))]
    prompts_demo4 = [[{"role": "user", "content": prefix_QA.format(demo4_captions['web'], demo4_captions['caption_1'], demo4_captions['caption_2'], demo4_captions['visual_entities'], questions[ix])}, {"role": "assistant", "content": demo4_answers[ix]}] for ix in range(len(questions))]
    prompts_demo5 = [[{"role": "user", "content": prefix_QA.format(demo5_captions['web'], demo5_captions['caption_1'], demo5_captions['caption_2'], demo5_captions['visual_entities'], questions[ix])}, {"role": "assistant", "content": demo5_answers[ix]}] for ix in range(len(questions))]
    prompts_demo6 = [[{"role": "user", "content": prefix_QA.format(demo6_captions['web'], demo6_captions['caption_1'], demo6_captions['caption_2'], demo6_captions['visual_entities'], questions[ix])}, {"role": "assistant", "content": demo6_answers[ix]}] for ix in range(len(questions))]
    prompts_demo7 = [[{"role": "user", "content": prefix_QA.format(demo7_captions['web'], demo7_captions['caption_1'], demo7_captions['caption_2'], demo7_captions['visual_entities'], questions[ix])}, {"role": "assistant", "content": demo7_answers[ix]}] for ix in range(len(questions))]
    prompts_demo8 = [[{"role": "user", "content": prefix_QA.format(demo8_captions['web'], demo8_captions['caption_1'], demo8_captions['caption_2'], demo8_captions['visual_entities'], questions[ix])}, {"role": "assistant", "content": demo8_answers[ix]}] for ix in range(len(questions))]
    
    all_prompt_demos = [prompts_demo1, prompts_demo2, prompts_demo3, prompts_demo4, prompts_demo5, prompts_demo6, prompts_demo7, prompts_demo8]

    demo_messages = []
    for i in range(len(all_prompt_demos)):
        for j in range(len(all_prompt_demos[i])):
            demos = [q[j] for q in all_prompt_demos]
            messages = [{"role": "system", "content": prefix_all}] + [item for sublist in demos for item in sublist]
            demo_messages.append(messages)

    return demo_messages

def generate_QA_demos_no_caption():
    prefix_all = """    You are a helpful assistant that answers questions about an image based on automatically generated captions and visual entities.
    You are given two automatically generated captions. The first caption provides a global description of the image, sometimes including people, but without naming them. The second caption provides more details about the people in the image, including their names and short biographies when that information is available. The two captions are a reliable source of information about the image.
    In some cases, you are given additional automated captions that focus on specific items present in the image.
    You are also given visual entities which may be relevant to the image, without certitude.
    Here are a few examples:"""
    prefix_QA = """
    Caption 1 - Global description: {}
    Caption 2 - More details about the people in the image: {} {extra_captions}
    Visual entities that may be relevant, without certitude: {}
    Question: {} Answer in one sentence. Answer with one word ("unknown") if the information is not provided.
    Answer: """

    people_Q = "Who is shown in the image?"
    apo_Q = "Which animal, plant, building, or object are shown in the image?"
    event_Q = "Which event is depicted in the image?"
    date_Q = "When was the image taken?"
    location_Q = "Where was the image taken?"
    motivation_Q = "Why was the image taken?"

    questions = [people_Q, apo_Q, event_Q, date_Q, location_Q, motivation_Q]

    demo1_captions = {
        'visual_entities': "Education, Lebanon, Teacher, Institution, Seminar, Presentation, Google Classroom, Organization, Amal Clooney, Middle East",
        'caption_2': "A teacher and students in a classroom setting, with the teacher demonstrating a mathematical concept on a whiteboard. One of the students is wearing a shirt with SABIS written on it.",
        'caption_1': "A classroom setting with a teacher and students, a whiteboard with mathematical equations, and a large screen displaying a math lesson.",
        'extra_captions': ""
    }
    demo1_answers = [
        "A group of primary school students. One of the students is wearing a shirt with SABIS written on it.",
        "A classroom with a white board.",
        "A lesson, likely at a SABIS school.",
        "Unknown.",
        "Likely a SABIS school.",
        "To report on classroom activities, likely at a SABIS school."
    ]

    demo2_captions = {
        'visual_entities': "American football, Team sport, Championship, Stadium, Competition, Multi-sport event, Player, Tournament, Team, Sports venue, Tourism",
        'caption_2': "A group of football players of the Washington Redskins, including a quarterback, who are lined up on a field, preparing for a play.",
        'caption_1': "A group of football players on a field, with a water tower in the background.",
        'extra_captions': """\nCaption 3 - More details about the buildings in the image: There is a building in the background but it is not possible to recognize it precisely.
        Caption 4 - More details about the sports in the image: The image shows American football players. Based on their outfits, they are part of the Washington Redskins team.\n"""
    }
    demo2_answers = [
        "American football players of the Washington Redskins.",
        "A training field.",
        "A training session of the Washington Redskins.",
        "Unknown.",
        "Unknown.",
        "To report on a training session of the Washington Redskins."
    ]

    demo3_captions = {
        'visual_entities': "Marcus Antonius, Public speaking, Motivational speaker, Speech, Spokesperson, Orator, Motivation, Public",
        'caption_2': "Mitt Romney, the Republican candidate for the presidential elections of 2012.",
        'caption_1': "A man at a podium with a microphone. In the background, there is a blurred American flag, which may indicate that the setting is a political event in the United States.",
        'extra_captions': ""
    }
    demo3_answers = [
        "Mitt Romney.",
        "A podium with a microphone. A blurred American flag in the background.",
        "Likely a political rally.",
        "Likely during the 2012 presidential campaign.",
        "Unknown",
        "Likely to report on a political rally of Mitt Romney during the presidential campaign of 2012."
    ]

    demo4_captions = {
        'visual_entities': "Profession",
        'caption_2': "Hillary Clinton, former Secretary of State, and Democrat candidate for the presidential elections of 2016.",
        'caption_1': "A woman standing to the right of the frame. Behind her is the American flag, indicating the setting is likely in the United States and may be of a formal or political nature.",
        'extra_captions': """\nCaption 3 -More details about the flags in the image: There is an United States flag in the image.\n"""
    }
    demo4_answers = [
        "Hillary Clinton.",
        "An American flag.",
        "Likely a political event related to the 2016 presidential elections.",
        "Likely during the 2016 presidential campaign.",
        "Unknown.",
        "Likely to report on a political event related to the 2016 presidential elections."
    ]

    demo5_captions = {
        'visual_entities': "France, Mali, Mali War, United States, NATO, Government of France, Civilian, Afribone, Guerrilla warfare, François Hollande",
        'caption_2': "There are two soldiers and they are searching a man. The people in the image cannot be identified based on the image content.",
        'caption_1': "The image shows two soldiers of an unidentified african country arresting and searching a man on the side of a road.",
        'extra_captions': ""
    }
    demo5_answers = [
        "Two soldiers and an arrested man.",
        "Unknown.",
        "Two soldiers are searching a man on the side of a road in Mali.",
        "Unknown.",
        "Mali.",
        "To report on two soldiers arresting and searching a man on the side of a road."
    ]

    demo6_captions = {
        'visual_entities': "Infantry, Troop, Marines, Woodland, Marksman, Military police, Biome, Reconnaissance, Military engineer, Mercenary, Tree",
        'caption_2': "The person in the image cannot be identified based on the image content.",
        'caption_1': " The image shows a man wearing a hat and a face mask,mstanding on the back of his pick-up. He is in front of a garage. A large US flag is hanging on the garage wall.",
        'extra_captions': """\nCaption 3 - More details about the vehicles in the image: The image shows the back of a pick-up. It is not possible to recognize the model based on the image.
        Caption 4 - More details about the flags in the image: The image shows a US flag hanging on the garage wall.\n"""
    }
    demo6_answers = [
        "An unknown man wearing a hat and a face mask.",
        "A pick-up, a garage, a US flag hanging on the garage wall.",
        "The man is standing on the back of his pick-up, next to his garage.",
        "Unknown.",
        "A garage in the United States.",
        "Unknown."
    ]

    demo7_captions = {
        'visual_entities': "PlayStation VR, Oculus Rift, Virtual reality, Virtual Reality Headset, Reality, Sony PlayStation, Elite Dangerous, Oculus Quest All-in-one Vr Gaming Headset 64gb, Oculus Quest, Oculus VR, Future, Headset",
        'caption_2': "The person in the image cannot be identified based on the image content.",
        'caption_1': "The image shows a woman playing a virtual reality game using the Playstation VR headset and move controllers.",
        'extra_captions': ""
    }
    demo7_answers = [
        "An unknown woman.",
        "A Playstation VR headset and move controllers.",
        "The woman is playing a Playstation VR game with a headset and move controllers.",
        "Unknown.",
        "Unknown.",
        "To report on the Playstation VR."
    ]

    demo8_captions = {
        'visual_entities': "Machine, Black and white, Gun, Monochrome / M, Cinematographer, Gun, Black, Physics, Simple machine, Science",
        'caption_2': "The people in the image cannot be identified.",
        'caption_1': "The image depicts a construction worker's helmet and safety gear hanging on a stand or post outside a store. In the background, there is a street scene with pedestrians and buildings, captured in black and white.",
        'extra_captions': ""
    }
    demo8_answers = [
        "Unknown.",
        "A construction worker's helmet and safety gear hanging on a stand or post outside a store.",
        "Unknown.",
        "Unknown.",
        "Unknown.",
        "Unknown."
    ]

    prompts_demo1 = [[{"role": "user", "content": prefix_QA.format(demo1_captions['caption_1'], demo1_captions['caption_2'], demo1_captions['visual_entities'], questions[ix], extra_captions=demo1_captions['extra_captions'])}, {"role": "assistant", "content": demo1_answers[ix]}] for ix in range(len(questions))]
    prompts_demo2 = [[{"role": "user", "content": prefix_QA.format(demo2_captions['caption_1'], demo2_captions['caption_2'], demo2_captions['visual_entities'], questions[ix], extra_captions=demo2_captions['extra_captions'])}, {"role": "assistant", "content": demo2_answers[ix]}] for ix in range(len(questions))]
    prompts_demo3 = [[{"role": "user", "content": prefix_QA.format(demo3_captions['caption_1'], demo3_captions['caption_2'], demo3_captions['visual_entities'], questions[ix], extra_captions=demo3_captions['extra_captions'])}, {"role": "assistant", "content": demo3_answers[ix]}] for ix in range(len(questions))]
    prompts_demo4 = [[{"role": "user", "content": prefix_QA.format(demo4_captions['caption_1'], demo4_captions['caption_2'], demo4_captions['visual_entities'], questions[ix], extra_captions=demo4_captions['extra_captions'])}, {"role": "assistant", "content": demo4_answers[ix]}] for ix in range(len(questions))]
    prompts_demo5 = [[{"role": "user", "content": prefix_QA.format(demo5_captions['caption_1'], demo5_captions['caption_2'], demo5_captions['visual_entities'], questions[ix], extra_captions=demo5_captions['extra_captions'])}, {"role": "assistant", "content": demo5_answers[ix]}] for ix in range(len(questions))]
    prompts_demo6 = [[{"role": "user", "content": prefix_QA.format(demo6_captions['caption_1'], demo6_captions['caption_2'], demo6_captions['visual_entities'], questions[ix], extra_captions=demo6_captions['extra_captions'])}, {"role": "assistant", "content": demo6_answers[ix]}] for ix in range(len(questions))]
    prompts_demo7 = [[{"role": "user", "content": prefix_QA.format(demo7_captions['caption_1'], demo7_captions['caption_2'], demo7_captions['visual_entities'], questions[ix], extra_captions=demo7_captions['extra_captions'])}, {"role": "assistant", "content": demo7_answers[ix]}] for ix in range(len(questions))]
    prompts_demo8 = [[{"role": "user", "content": prefix_QA.format(demo8_captions['caption_1'], demo8_captions['caption_2'], demo8_captions['visual_entities'], questions[ix], extra_captions=demo8_captions['extra_captions'])}, {"role": "assistant", "content": demo8_answers[ix]}] for ix in range(len(questions))]

    all_prompt_demos = [prompts_demo1, prompts_demo2, prompts_demo3, prompts_demo4, prompts_demo5, prompts_demo6, prompts_demo7, prompts_demo8]

    demo_messages = []
    for i in range(len(all_prompt_demos)):
        for j in range(len(all_prompt_demos[i])):
            demos = [q[j] for q in all_prompt_demos]
            messages = [{"role": "system", "content": prefix_all}] + [item for sublist in demos for item in sublist]
            demo_messages.append(messages)

    return demo_messages

def generate_veracity_demo_with_captions():

    prefix_reason_all = """You are a helpful assistant that verifies if a caption is accurate for an image or if it is an attempt at out-of-context misinformation.
    Instead of the image itself, you are given structured context information about the image. You need to verify whether the caption is supported by the context information.
    Here are a few examples:"""

    reason_prompt = """
    Context information:
    {}
    """

    prompt1 = """Things: A neon horse. 
    Event: The Chinese new year celebration for the Year of the Horse.
    Date: The image was taken for the celebrations of the start of the Year of the Horse. The last Year of the Horse started on January 31, 2014. Hence, the date is likely January 31, 2014.
    Location: Fo Guang Shan Dong Zen temple in Klang, Malaysia.
    Motivation: To report on the celebrations for the start of the Year of the Horse in Malaysia.
    Source: The Guardian.

    Caption to verify: Cirque du Soleil latest act to boycott North Carolina.
    Given the context information, is the caption accurate or is it out-of-context? If there are too many unknown elements to provide a clear decision, you can answer that the accuracy of the caption is “unknown”, potentially leaning more towards accurate or out-of-context. Provide a detailed reasoning. Then provide your answer strictly among the following choices : “accurate”, “unknown, probably accurate”, “unknown”, “unknown, probably out-of-context”, “out-of-context”."""
    reason1  = """Reasoning: The caption claims that the image is related to Cirque du soleil. Cirque du soleil is a Canadian organization from Quebec with no direct connection to Chinese new year celebrations. Furthermore, the caption claims that the image is related to North Carolina. That is incorrect and contradicts the context (location : Malaysia).
    Answer: Out-of-context."""

    prompt2 = """People: Donald Trump.
    Event: A campaign rally of Donald Trump for the Republican presidential primary of 2015.
    Date: December 29, 2015.
    Location: Council Bluffs, Iowa, USA.
    Motivation: To report on a political campaign rally of Donald Trump in the context of the Republican presidential primary.

    Caption to verify: Donald Trump told a campaign rally in Council Bluffs Iowa I don't want to take any chances.
    Given the context information, is the caption accurate or is it out-of-context? If there are too many unknown elements to provide a clear decision, you can answer that the accuracy of the caption is “unknown”, potentially leaning more towards accurate or out-of-context. Provide a detailed reasoning. Then provide your answer strictly among the following choices : “accurate”, “unknown, probably accurate”, “unknown”, “unknown, probably out-of-context”, “out-of-context”."""
    reason2 = """Reasoning: The caption claims that the image shows Donald Trump. That is supported by the context (people : Donald Trump). The caption claims that the image is located in Council Bluffs, Iowa. That is supported by the context (location : Councils Bluffs, Iowa, USA). The caption claims that the image is related to a campaign rally. That is supported by the context (event : a campaign rally).
    Answer: Accurate."""

    prompt3 = """People: A cyclist.
    Things: The Black Cultural Archives building.
    Location: London. More specifically, the Black Cultural Archives are located in Brixton, London, England.
    Source: The Guardian.

    Caption to verify: The Black Cultural Archives sits in the heart of Brixton just five minutes from its original home.
    Given the context information, is the caption accurate or is it out-of-context? If there are too many unknown elements to provide a clear decision, you can answer that the accuracy of the caption is “unknown”, potentially leaning more towards accurate or out-of-context. Provide a detailed reasoning. Then provide your answer strictly among the following choices : “accurate”, “unknown, probably accurate”, “unknown”, “unknown, probably out-of-context”, “out-of-context”."""
    reason3 = """Reasoning: The caption claims that the image shows the Black Cultural Archives. That is correct and supported by the context (object : Black Cultural Archives). The caption claims that the Black Cultural Archives sits just five minutes from its original home. This is a claim. It cannot be verified but it is not relevant to assess whether or not the caption is accurate for the image.
    Answer: Accurate."""

    prompt4 = """Things: Trees, a street. A bright light in the background, which could indicate a fire.
    Event: A home was burned down at Satellite Beach during Hurricane Matthew.
    Date: Hurricane Matthew happened in October 2016.
    Location: Satellite Beach, which is located in Florida, USA.
    Motivation: To report on the damages caused by Hurricane Matthew in Florida.
    Source: Florida Today.

    Caption to verify: Enoch Kinard looks through his destroyed house on Ellerslie Avenue in LaPlace La.
    Given the context information, is the caption accurate or is it out-of-context? If there are too many unknown elements to provide a clear decision, you can answer that the accuracy of the caption is “unknown”, potentially leaning more towards accurate or out-of-context. Provide a detailed reasoning. Then provide your answer strictly among the following choices : “accurate”, “unknown, probably accurate”, “unknown”, “unknown, probably out-of-context”, “out-of-context”."""
    reason4 = """Reasoning: The caption claims that the image shows a house in LaPlace, Louisiana (La). That is incorrect and contradicts the context (location : Satellite Beach, Florida).
    Answer: Out-of-context."""

    prompt5 = """People: Novak Djokovic.
    Things: A tennis court, a tennis racket.
    Event: Novak Djokovic jumping on the court during a game with Rafael Nadal at the US Open 2013.
    Date: September 9, 2013.
    Location: US Open at USTA Billie Jean King National Tennis Center in Flushing Meadows, Queens, New York City.
    Motivation: To report on the final of the US Open 2013 opposing Novak Djokovic to Rafael Nadal.

    Caption to verify: Novak Djokovic of Serbia jumps during his singles semifinal match against Juan Martin Del Potro He won in five sets.
    Given the context information, is the caption accurate or is it out-of-context? If there are too many unknown elements to provide a clear decision, you can answer that the accuracy of the caption is “unknown”, potentially leaning more towards accurate or out-of-context. Provide a detailed reasoning. Then provide your answer strictly among the following choices : “accurate”, “unknown, probably accurate”, “unknown”, “unknown, probably out-of-context”, “out-of-context”."""
    reason5 = """Reasoning: The caption claims that the image shows Novak Djokovic of Serbia. That is correct and supported by the context (people : Novak Djokovic). The caption claims that the image shows a game between Djokovic and Del Potro. That is incorrect and contradicts the context (event : a game between Djokovic and Nadal at the US Open).
    Answer: Out-of-context."""

    prompt6 = """People: Sourena, an LGBTQ asylum seeker in the Mideast.
    Things: A kitchen, a table with a mirror, a cigarette, a bathrobe.
    Event: Sourena is smoking in his kitchen.
    Location: Isparta, Western Turkey.
    Motivation: To report on the life of Sourena, an LGBTQ asylum seeker who left Iran and applied for resettlement in Canada via the UNHCR
    Source: Washington Post.

    Caption to verify: Sourena in his kitchen in Isparta Sourena left Iran and applied for resettlement in Canada via the UNHCR on sexuality grounds He was successful and now lives in Toronto.
    Given the context information, is the caption accurate or is it out-of-context? If there are too many unknown elements to provide a clear decision, you can answer that the accuracy of the caption is “unknown”, potentially leaning more towards accurate or out-of-context. Provide a detailed reasoning. Then provide your answer strictly among the following choices : “accurate”, “unknown, probably accurate”, “unknown”, “unknown, probably out-of-context”, “out-of-context”."""
    reason6 = """Reasoning: The caption claims that the image shows Sourena in his kitchen. That is supported by the context (event : Sourena smoking in his kitchen). The caption claims that the image was taken in Isparta. That is supported by the context (location : Isparta, Western Turkey). The caption claims that Sourena was successful in his application for resettlement in Canada. That is not mentioned in the context but it does not impact the accuracy of the image.
    Answer: Accurate."""

    prompt7 = """People: Billy Sharp of Reading and a team mate, and 2 players of Nottingham Forest.
    Things: A football field.
    Event: A football game between Reading and Nottingham Forest, likely during the Championship 2011-2012 season.
    Date: Championship 2011-2012.
    Motivation: To report on an action involving Billy Sharp and other players during a game between Reading and Nottingham Forest.
    Source: The Guardian.

    Caption to verify: Mike Williamson challenges Johan Elmander.
    Given the context information, is the caption accurate or is it out-of-context? If there are too many unknown elements to provide a clear decision, you can answer that the accuracy of the caption is “unknown”, potentially leaning more towards accurate or out-of-context. Provide a detailed reasoning. Then provide your answer strictly among the following choices : “accurate”, “unknown, probably accurate”, “unknown”, “unknown, probably out-of-context”, “out-of-context”."""
    reason7 = """Reasoning: The caption claims that the image shows Mike Williamson and Johan Elmander. That is incorrect and contradicts the context (people : Billy Sharp of Reading and a team mate, and 2 players of Nottingham Forest). Mike Williamson or Johan Elmander are football players, but neither of them have played for Reading or Nottingham Forest.
    Answer: Out-of-context."""

    prompt8 = """Things: The entrance of a building with the HSBC logo.
    Date: March 1, 2010.
    Location: London.

    Caption to verify: HSBC bank Chairman Douglas Flint says a decision on whether to relocate away from the City will be taken this year.
    Given the context information, is the caption accurate or is it out-of-context? If there are too many unknown elements to provide a clear decision, you can answer that the accuracy of the caption is “unknown”, potentially leaning more towards accurate or out-of-context. Provide a detailed reasoning. Then provide your answer strictly among the following choices : “accurate”, “unknown, probably accurate”, “unknown”, “unknown, probably out-of-context”, “out-of-context”."""
    reason8 = """Reasoning: The caption claims that the image is related to the HSBC bank. That is supported by the context (things : the entrance of a building with the HSBC logo). The caption contains a statement by Chairman Douglas Flint. This statement is not relevant to decide whether the caption is accurate or out-of-context.
    Answer: Accurate."""

    prompts_list = [prompt1, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7, prompt8]
    reasons_list = [reason1, reason2, reason3, reason4, reason5, reason6, reason7, reason8]

    all_prompts_no_reason = [{"role": "user", "content": reason_prompt.format(p)} for p in prompts_list]
    all_reason_prompts = [{"role": "assistant", "content": reason} for reason in reasons_list]
    prompts_with_reasons = [[p, r] for p, r in zip(all_prompts_no_reason, all_reason_prompts)]
    all_reason_prompts = [{"role": "system", "content": prefix_reason_all}] + [item for sublist in prompts_with_reasons for item in sublist]

    return all_reason_prompts

def generate_veracity_demo_no_captions():

    prefix_reason_all = """You are a helpful assistant that verifies if a caption is accurate for an image or if it is an attempt at out-of-context misinformation.
    Instead of the image itself, you are given structured context information about the image. You need to verify whether the caption is supported by the context information.
    Here are a few examples:"""

    reason_prompt = """
    Context information:
    {}
    """

    prompt1 = """ People: A group of primary school students.
    Things: A classroom with a white board.
    Event: A lesson, likely at a school.
    Location: Likely a school.
    Motivation: To report on classroom activities, likely at a school.

    Caption to verify: Primary school children at a SABIS school in Lebanon.
    Given the context information, is the caption accurate or is it out-of-context? If there are too many unknown elements to provide a clear decision, you can answer that the accuracy of the caption is “unknown”, potentially leaning more towards accurate or out-of-context. Provide a detailed reasoning. Then provide your answer strictly among the following choices : “accurate”, “unknown, probably accurate”, “unknown”, “unknown, probably out-of-context”, “out-of-context”."""
    reason1  = """Reasoning: The caption claims that the image shows primary school children. That is correct (people: A group of primary school students). The caption claims that it is a SABIS school. That cannot be confirmed by the context information. The caption claims that the image was taken in Lebanon. This is neither supported or refuted by the context (location: Likely a school). There is not enough information to confirm that the caption is accurate.
    Answer: Unknown, probably out-of-context."""

    prompt2 = """People: American football players of the Washington Redskins.
    Things: A training field.
    Event: A training session of the Washington Redskins.
    Motivation: To report on a training session of the Washington Redskins.
    
    Caption to verify: The Vikings will play for two years at the University of Minnesota's new stadium.
    Given the context information, is the caption accurate or is it out-of-context? If there are too many unknown elements to provide a clear decision, you can answer that the accuracy of the caption is “unknown”, potentially leaning more towards accurate or out-of-context. Provide a detailed reasoning. Then provide your answer strictly among the following choices : “accurate”, “unknown, probably accurate”, “unknown”, “unknown, probably out-of-context”, “out-of-context”."""
    reason2 = """Reasoning: The caption claims that the image is related to the Minnesota Vikings. That is incorrect and contradicts the context (people : Washington Redskins).
    Answer: Out-of-context."""

    prompt3 = """People: Mitt Romney.
    Things: A podium with a microphone. A blurred American flag in the background.
    Event: Likely a political rally.
    Date: Likely during the 2012 presidential campaign.
    Motivation: Likely to report on a political rally of Mitt Romney during the presidential campaign of 2012.

    Caption to verify: Mitt Romney speaks during a campaign stop in Reno.
    Given the context information, is the caption accurate or is it out-of-context? If there are too many unknown elements to provide a clear decision, you can answer that the accuracy of the caption is “unknown”, potentially leaning more towards accurate or out-of-context. Provide a detailed reasoning. Then provide your answer strictly among the following choices : “accurate”, “unknown, probably accurate”, “unknown”, “unknown, probably out-of-context”, “out-of-context”."""
    reason3 = """Reasoning: The caption claims that the image shows Mitt Romney. That is supported by the context (people : Mitt Romney). The caption claims that the image shows a campaign stop. That is likely true given the context information. The caption claims that the image was taken in Reno. This cannot be verified based on the context information. Given that the rest of the caption is accurate, and that Reno is a major town in Nevada, this is very likely to be accurate too.
    Answer: Unknown, probably accurate."""

    prompt4 = """People: Hillary Clinton.
    Things: An American flag.
    Event: Likely a political event related to the 2016 presidential elections.
    Date: Likely during the 2016 presidential campaign.
    Motivation: Likely to report on a political event related to the 2016 presidential elections.

    Caption to verify: Secretary of State Hillary Rodham Clinton is hospitalized in New York with a blood clot She is expected to stay in the hospital for 48 hours.
    Given the context information, is the caption accurate or is it out-of-context? If there are too many unknown elements to provide a clear decision, you can answer that the accuracy of the caption is “unknown”, potentially leaning more towards accurate or out-of-context. Provide a detailed reasoning. Then provide your answer strictly among the following choices : “accurate”, “unknown, probably accurate”, “unknown”, “unknown, probably out-of-context”, “out-of-context”."""
    reason4 = """Reasoning: The caption claims that the image shows Hillary Rodham Clinton. That is supported by the context (people: Hillary Clinton). The caption claims that the image is related to the hospitalization of Hillary Clinton. That is incorrect and contradicts the context (event: a political event). 
    Answer: Out-of-context."""

    prompt5 = """People: Two soldiers and an arrested man.
    Event: Two soldiers are searching a man on the side of a road in Mali.
    Location: Mali.
    Motivation: To report on two soldiers arresting and searching a man on the side of a road.

    Caption to verify: Malian soldiers search a man at a bridge in Markala.
    Given the context information, is the caption accurate or is it out-of-context? If there are too many unknown elements to provide a clear decision, you can answer that the accuracy of the caption is “unknown”, potentially leaning more towards accurate or out-of-context. Provide a detailed reasoning. Then provide your answer strictly among the following choices : “accurate”, “unknown, probably accurate”, “unknown”, “unknown, probably out-of-context”, “out-of-context”."""
    reason5 = """Reasoning: The caption claims that the image shows soldiers searching a man. That is supported by the context (event : two soldiers are searching a man). The caption claims that the soldiers are Malian. That is partially supported by the context (location : Mali). The caption claims that the image was taken near a bridge in Markala. That is partially supported by the context, as Markala is a town in Mali (location : Mali). 
    Answer: Unknown, probably accurate."""

    prompt6 = """People: An unknown man wearing a hat and a face mask.
    Things: A pick-up, a garage, a US flag hanging on the garage wall.
    Event: The man is standing on the back of his pick-up, next to his garage.
    Location: A garage in the United States.

    Caption to verify: Aaron Humprey of the El Dorado Hot Shots watches the Lake Fire burn along Highway 38 in the San Bernardino National Forest.
    Given the context information, is the caption accurate or is it out-of-context? If there are too many unknown elements to provide a clear decision, you can answer that the accuracy of the caption is “unknown”, potentially leaning more towards accurate or out-of-context. Provide a detailed reasoning. Then provide your answer strictly among the following choices : “accurate”, “unknown, probably accurate”, “unknown”, “unknown, probably out-of-context”, “out-of-context”."""
    reason6 = """Reasoning: The caption claims that a man named Aaron Humprey watches the Lake Fire burn. That is incorrect and contradicts the context (event : The man is standing on the back of his pick-up, next to his garage). The caption claims that the image was taken along a highway. That is incorrect and contradicts the context (location : a garage). 
    Answer: Out-of-context."""

    prompt7 = """People: An unknown woman.
    Things: A Playstation VR headset and move controllers.
    Event: The woman is playing a Playstation VR game with a headset and move controllers.
    Motivation: To report on the Playstation VR.

    Caption to verify: An upgraded PS4 would benefit its VR experience.
    Given the context information, is the caption accurate or is it out-of-context? If there are too many unknown elements to provide a clear decision, you can answer that the accuracy of the caption is “unknown”, potentially leaning more towards accurate or out-of-context. Provide a detailed reasoning. Then provide your answer strictly among the following choices : “accurate”, “unknown, probably accurate”, “unknown”, “unknown, probably out-of-context”, “out-of-context”."""
    reason7 = """Reasoning: The caption claims to be related to Playstation and VR. That is correct and supported by the context (motivation : to report on the Playstation VR). The caption claims that an upgraded PS4 would benefit the VR experience. That is an opinion. It is not relevant to decide whether the caption is accurate or out-of-context. 
    Answer: Accurate."""

    prompt8 = """Things: A construction worker's helmet and safety gear hanging on a stand or post outside a store.

    Caption to verify: Field Cycles make 15 bikes a year at their workshop in Sheffield.
    Given the context information, is the caption accurate or is it out-of-context? If there are too many unknown elements to provide a clear decision, you can answer that the accuracy of the caption is “unknown”, potentially leaning more towards accurate or out-of-context. Provide a detailed reasoning. Then provide your answer strictly among the following choices : “accurate”, “unknown, probably accurate”, “unknown”, “unknown, probably out-of-context”, “out-of-context”."""
    reason8 = """Reasoning: The caption claims that the image is related to the Field Cycles workshop in Sheffield. That  cannot be verified based on the context. However, the context mentions a construction worker’s helmet, which is unlikely to be associated with a Field Cycles workshop. 
    Answer: Unknown, probably out-of-context."""

    prompts_list = [prompt1, prompt2, prompt3, prompt4, prompt5, prompt6, prompt7, prompt8]
    reasons_list = [reason1, reason2, reason3, reason4, reason5, reason6, reason7, reason8]

    all_prompts_no_reason = [{"role": "user", "content": reason_prompt.format(p)} for p in prompts_list]
    all_reason_prompts = [{"role": "assistant", "content": reason} for reason in reasons_list]
    prompts_with_reasons = [[p, r] for p, r in zip(all_prompts_no_reason, all_reason_prompts)]
    all_reason_prompts = [{"role": "system", "content": prefix_reason_all}] + [item for sublist in prompts_with_reasons for item in sublist]

    return all_reason_prompts