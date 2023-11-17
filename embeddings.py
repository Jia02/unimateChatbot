contexts = ["Students are responsible for ensuring their attendance is rightly recorded via the mechanism provided by the Schools. Unsatisfactory Attendance Report (UAR) will be issued to students who do not meet the minimum attendance of 80%. It is the student's responsibility to inform the lecturer/Programme Leader/Coordinator of his/her absence. The student may be required to disclose the reasons for absence.",
            
           "A deferment is normally granted for 1 semester period. You are allowed a maximum of 2 deferments or a total deferment period of not more than 1 calendar year in your programme of study. It is your responsibility to be aware of the designated enrolment timeframe before applying for deferment in order to avoid paying any penalties. During the deferment period, you are considered a registered student of the University. You will be notified of the date to return to your programme of study and expected to report yourself for classes within the designated timeframe.",

           "Students shall be permitted to change subject enrolments within the first two (2) weeks of a long semester (14 weeks), or one (1) week of a short semester (7 weeks), which is known as the Subject Add/Drop period. Students are not permitted to register for additional subjects after the Subject Add/Drop period. Tuition fees will not be refunded for subjects withdrawn after the Subject Add/Drop period.",

           "Major Cheating in Examination Examples include: iii. Blatant use of written, printed or electronic material not permitted within the rubric of the examination. iv. Communication with any other student in the examination room.  v. Inappropriate communication with a member of academic staff during the period of the examination. vi. Obtaining unauthorised material prior to the examination. vii. Second minor incident.",

           "The action taken where academic malpractice has been found, and the severity of the penalty applied, will depend on the individual circumstances prevailing. Penalties that may be considered are: i. Warning in writing to the student (kept on the students file until the completion of their studies). ii. A requirement to resubmit the relevant piece(s) of work by a specific deadline as a First Attempt (mark not capped). iii. A requirement to resubmit the relevant piece(s) of work by a specific deadline as a Second Attempt capped at the minimum pass mark. iv. A requirement to resubmit the relevant piece(s) of work by a specific deadline as a Final Attempt capped at the minimum pass mark.",

           "I don't understand what you are asking. Please ask a complete question."]

keywords_map = {
        0: ["attendance", "attendances", "absence", "absences", "absent", "absented", "80%"],
        1: ["deferment", "deferments", "defer", "defers", "deferred"],
        2: ["add drop", "add/drop", "add / drop"],
        3: ["major", "cheating", "cheat", "cheated", "cheats"],
        4: ["malpractice"]
    }

def find_context(prompt):
    prompt = prompt.lower()

    # Check which category the prompt belongs to
    for index, keywords in keywords_map.items():
        if any(keyword in prompt for keyword in keywords):
            return contexts[index]

    # Set default context if none of the categories match
    return contexts[-1]
