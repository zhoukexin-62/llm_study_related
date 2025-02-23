# -*- coding: utf-8 -*-
# @Time    : 2025/2/14 17:03
# @Author  : Bin_NONG
# @FileName: data_prepare.py
# @Software: PyCharm
# @Affiliation: xxx
# here is list of 50 examples of prompt and completions focused on singing skills

# 所有的数据，需要自己进行读取


samples = [
    {
        "prompt": "Question 1: What is the first step to improving your singing voice?",
        "completion": "Answer 1: Begin by warming up your vocal cords with gentle exercises like humming or lip trills."
    },
    {
        "prompt": "Question 2: How can you improve your breath control while singing?",
        "completion": "Answer 2: Practice diaphragmatic breathing by inhaling deeply through your nose and controlling your exhale while singing sustained notes."
    },
    {
        "prompt": "Question 3: What are common mistakes beginners make when learning to sing?",
        "completion": "Answer 3: Common mistakes include straining the voice, neglecting to warm up, and improper breathing techniques."
    },
    {
        "prompt": "Question 4: How do you find your vocal range?",
        "completion": "Answer 4: Experiment by singing scales up and down to identify the lowest and highest note you can comfortably sustain."
    },
    {
        "prompt": "Question 5: What exercises help improve pitch accuracy?",
        "completion": "Answer 5: Practice scales, use a piano or digital tuner for reference, and try interval training to enhance pitch precision."
    },
    {
        "prompt": "Question 6: How can you protect your vocal cords from strain?",
        "completion": "Answer 6: Avoid over-singing, stay hydrated, and limit exposure to irritants like smoke or excessively dry air."
    },
    {
        "prompt": "Question 7: What role does posture play in singing?",
        "completion": "Answer 7: Proper posture opens up the diaphragm and throat, allowing for better breath support and vocal projection."
    },
    {
        "prompt": "Question 8: How do you improve vocal tone quality?",
        "completion": "Answer 8: Focus on resonance and placement, and practice vocal exercises like sirens to achieve a smoother tone."
    },
    {
        "prompt": "Question 9: What techniques help develop vibrato in singing?",
        "completion": "Answer 9: Practice sustaining notes while gently oscillating your pitch; focus on relaxation and consistent airflow."
    },
    {
        "prompt": "Question 10: How can you overcome stage fright while singing?",
        "completion": "Answer 10: Prepare thoroughly, practice mindfulness techniques, and start performing in smaller, supportive settings."
    },
    {
        "prompt": "Question 11: What warm-up exercises are essential for singers?",
        "completion": "Answer 11: Lip trills, humming, and singing scales are effective warm-up exercises that prepare your voice for singing."
    },
    {
        "prompt": "Question 12: How do you sing high notes without straining?",
        "completion": "Answer 12: Use proper breath support, relax your throat, and place the sound forward in your mask (resonance areas)."
    },
    {
        "prompt": "Question 13: What is the importance of ear training for singers?",
        "completion": "Answer 13: Ear training helps singers recognize pitches, harmonies, and intervals, improving overall musicality and pitch accuracy."
    },
    {
        "prompt": "Question 14: How do you maintain vocal health during a busy singing schedule?",
        "completion": "Answer 14: Stay hydrated, get adequate rest, and avoid overuse by incorporating regular vocal breaks and proper warm-ups."
    },
    {
        "prompt": "Question 15: What foods and drinks should singers avoid?",
        "completion": "Answer 15: Avoid dairy, caffeine, and spicy foods before singing, as they can cause mucus buildup or throat irritation."
    },
    {
        "prompt": "Question 16: How can you improve diction while singing?",
        "completion": "Answer 16: Practice enunciation exercises like tongue twisters and focus on shaping vowels and consonants clearly."
    },
    {
        "prompt": "Question 17: What are the benefits of working with a vocal coach?",
        "completion": "Answer 17: A vocal coach provides personalized feedback, helps identify areas for improvement, and teaches proper technique."
    },
    {
        "prompt": "Question 18: How do you transition smoothly between vocal registers?",
        "completion": "Answer 18: Practice blending your chest and head voice through exercises like sirens and scales to achieve a seamless transition."
    },
    {
        "prompt": "Question 19: What is the purpose of singing scales during practice?",
        "completion": "Answer 19: Singing scales improves pitch accuracy, strengthens the voice, and helps with transitioning between notes smoothly."
    },
    {
        "prompt": "Question 20: How do you develop confidence in your singing voice?",
        "completion": "Answer 20: Record and listen to your progress, focus on your strengths, and perform regularly to build confidence over time."
    },
    {
        "prompt": "Question 21: What is falsetto, and how do you practice it?",
        "completion": "Answer 21: Falsetto is a light, airy register above the chest voice. Practice by relaxing your throat and gently accessing higher notes."
    },
    {
        "prompt": "Question 22: Why is hydration important for singers?",
        "completion": "Answer 22: Hydration keeps the vocal cords lubricated, preventing strain and allowing for smoother vocal production."
    },
    {
        "prompt": "Question 23: How do you project your voice without shouting?",
        "completion": "Answer 23: Use proper breath support, open your throat, and focus on forward resonance to amplify your sound naturally."
    },
    {
        "prompt": "Question 24: What techniques can improve vocal agility?",
        "completion": "Answer 24: Practice fast-paced scales, arpeggios, and runs while maintaining clarity and control over each note."
    },
    {
        "prompt": "Question 25: How do you develop a unique singing style?",
        "completion": "Answer 25: Experiment with phrasing, dynamics, and tone to find a style that reflects your personality and musical influences."
    },
    {
        "prompt": "Question 26: What are the signs of vocal fatigue?",
        "completion": "Answer 26: Hoarseness, difficulty hitting notes, and a feeling of tightness in the throat are common signs of vocal fatigue."
    },
    {
        "prompt": "Question 27: How do you handle voice cracks while singing?",
        "completion": "Answer 27: Identify the cause, such as tension or lack of breath support, and practice transitioning between registers smoothly."
    },
    {
        "prompt": "Question 28: Why is it important to practice dynamics in singing?",
        "completion": "Answer 28: Dynamics add emotion and variety to your performance, making your singing more expressive and engaging."
    },
    {
        "prompt": "Question 29: How do you train for long singing sessions?",
        "completion": "Answer 29: Build endurance gradually by practicing short sessions and focusing on proper vocal technique and breath control."
    },
    {
        "prompt": "Question 30: What role does emotion play in singing?",
        "completion": "Answer 30: Emotion connects you with your audience and brings authenticity and depth to your performance."
    },
    {
        "prompt": "Question 31: How can you improve your ability to harmonize?",
        "completion": "Answer 31: Practice singing along with recordings, focus on intervals, and work with other singers to develop harmony skills."
    },
    {
        "prompt": "Question 32: What are the benefits of recording your singing practice?",
        "completion": "Answer 32: Recording allows you to analyze your tone, pitch, and phrasing, helping you identify areas for improvement."
    },
    {
        "prompt": "Question 33: How do you practice singing in different genres?",
        "completion": "Answer 33: Study the stylistic elements of each genre, such as phrasing and tone, and practice songs that challenge your versatility."
    },
    {
        "prompt": "Question 34: What exercises help with vocal endurance?",
        "completion": "Answer 34: Long-tone exercises, sustained scales, and practicing songs at various dynamic levels build vocal stamina over time."
    },
    {
        "prompt": "Question 35: How do you avoid nasal singing?",
        "completion": "Answer 35: Focus on proper resonance placement in your mouth and throat, and avoid excessive tension in your nasal passages."
    },
    {
        "prompt": "Question 36: What is the role of articulation in singing?",
        "completion": "Answer 36: Articulation ensures that lyrics are clear and understandable, enhancing the listener's experience."
    },
    {
        "prompt": "Question 37: How can you improve your rhythm as a singer?",
        "completion": "Answer 37: Practice with a metronome, clap rhythms, and sing along with instrumental tracks to improve timing and precision."
    },
    {
        "prompt": "Question 38: What are the challenges of singing in a group or choir?",
        "completion": "Answer 38: Challenges include blending with others, maintaining pitch, and following the conductor's cues for dynamics and timing."
    },
    {
        "prompt": "Question 39: How do you prepare for a live singing performance?",
        "completion": "Answer 39: Warm up thoroughly, rehearse your setlist, and visualize a successful performance to build confidence."
    },
    {
        "prompt": "Question 40: Why is it important to rest your voice after singing?",
        "completion": "Answer 40: Resting allows your vocal cords to recover, preventing strain and potential long-term damage."
    },
    {
        "prompt": "Question 41: How do you sing with emotion without losing technical accuracy?",
        "completion": "Answer 41: Balance emotion with technique by practicing expressive phrasing and ensuring consistent breath support."
    },
    {
        "prompt": "Question 42: What are the benefits of singing scales daily?",
        "completion": "Answer 42: Daily scales improve pitch accuracy, vocal agility, and overall strength, keeping your voice in top condition."
    },
    {
        "prompt": "Question 43: How do you recover from vocal strain or overuse?",
        "completion": "Answer 43: Rest your voice, stay hydrated, and avoid whispering, which can further strain your vocal cords."
    },
    {
        "prompt": "Question 44: How can you improve your stage presence as a singer?",
        "completion": "Answer 44: Engage with your audience, use natural gestures, and maintain eye contact to enhance your connection with listeners."
    },
    {
        "prompt": "Question 45: What are signs that you need professional vocal training?",
        "completion": "Answer 45: Persistent pitch issues, vocal fatigue, or difficulty with certain techniques are signs that professional training may help."
    },
    {
        "prompt": "Question 46: How do you maintain vocal consistency across performances?",
        "completion": "Answer 46: Stick to a regular practice routine, warm up thoroughly, and monitor your vocal health to ensure consistent performance."
    },
    {
        "prompt": "Question 47: How can you expand your vocal range safely?",
        "completion": "Answer 47: Gradually practice exercises that stretch your range, like sirens, without pushing or straining your voice."
    },
    {
        "prompt": "Question 48: What is the importance of storytelling in singing?",
        "completion": "Answer 48: Storytelling connects you with the audience, making your performance more impactful and memorable."
    },
    {
        "prompt": "Question 49: How do you handle feedback as a singer?",
        "completion": "Answer 49: Listen to feedback with an open mind, focus on constructive criticism, and use it as an opportunity to grow."
    },
    {
        "prompt": "Question 50: What is the final step in preparing for a vocal performance?",
        "completion": "Answer 50: Do a final warm-up, review your setlist, and focus on calming your nerves before stepping onto the stage."
    }
]

