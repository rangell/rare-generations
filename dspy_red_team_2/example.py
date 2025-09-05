# https://x.com/MaximeRivest/article/1948024214763548883
import dspy

import re

def is_french(text):
    # Naive French detector: check for common French words/accents
    french_markers = [
        r"\b(le|la|les|un|une|des|du|de|et|à|est|sont|avec|pour|sur|par|mais|ou|où|que|qui|quand|comment|nous|vous|ils|elles|ça|ce|cette|ces)\b",
        r"[éèêàùçîôâœëïü]",
    ]
    return any(re.search(marker, text.lower()) for marker in french_markers)

def translation_judge(example, prediction, trace=None):
    """
    Return 1.0 if the output looks French, else 0.0.
    Doing the cast explicitly guarantees we never hand DSPy a None.
    """
    output = prediction.get("generation", "") or ""
    try:
        return float(is_french(output))
    except Exception:
        # Anything weird is just a miss
        return 0.0

# Define the SimplestAdapter as before
class SimplestAdapter(dspy.Adapter):
    def __call__(self, lm, lm_kwargs, signature, demos, inputs):
        # print(inputs)
        system_content = signature.instructions
        print('btw this is the system content', system_content)
        if demos:
            system_content
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": inputs["prompt"]},
        ]
        outputs = lm(messages=messages, **lm_kwargs)
        return [{"generation": outputs[0]}]

# Do NOT call dspy.configure(adapter=SimplestAdapter())
# Subclass Predict to use the custom adapter only for this instance
class MyPredict(dspy.Predict):
    def forward(self, **kwargs):
        adapter = SimplestAdapter()
        print('my predict forward, signature is', self.signature.instructions)
        with dspy.settings.context(adapter=adapter):
            return super().forward(**kwargs)


def format_demos(demos):
    """
    Wrap every demo once – no duplicated header lines.
    """
    parts = ["Here are examples of your expected behavior.",
             "<examples>"]
    for i, demo in enumerate(demos, 1):
        parts += [
            f"<example_{i}>",
            "User:",
            demo["prompt"],
            "Assistant:",
            demo["generation"],
            f"</example_{i}>",
        ]
    parts.append("</examples>")
    return "\n".join(parts)

def optimize(
    *,
    training_inputs: list[str],
    training_outputs: list[str],
    llm_judge,
    system_prompt: str = "",
    max_few_shots: int = 0,
    teacher_model=None,
    student_model=None,
):
    """
    One‑stop helper that (1) turns parallel input / output lists into a DSPy
    training‑set, (2) builds / optimises a tiny translation programme, and
    (3) returns the auto‑generated system‑prompt (with optional few‑shot
    examples baked‑in).

    Parameters
    ----------
    training_inputs, training_outputs : list[str]
        Parallel lists of user prompts and the desired assistant replies.
    llm_judge : str | Callable
        Either a *string* with judging instructions **or** a fully‑formed
        `metric(example, prediction, trace)->float` callable.
    system_prompt : str, optional
        A starting prompt to improve upon (default empty).
    max_few_shots : int, optional
        Upper‑bound on examples the optimiser may add to the prompt.
    teacher_model, student_model : str | dspy.LM | None
        Identifiers *or* `dspy.LM` objects.  If only one is given, we fall
        back gracefully to the globally configured LM.

    Returns
    -------
    str
        The final system‑prompt text, ready to feed any chat‑completion API.
    """

    # ------------------------------------------------------------------ #
    # 0 .  Basic validation                                              #
    # ------------------------------------------------------------------ #
    if len(training_inputs) != len(training_outputs):
        raise ValueError("`training_inputs` and `training_outputs` must "
                         "have the same length.")

    # ------------------------------------------------------------------ #
    # 1 .  Build the training set                                        #
    # ------------------------------------------------------------------ #
    examples = [
        dspy.Example(prompt=inp, generation=out)
        for inp, out in zip(training_inputs, training_outputs, strict=True)
    ]
    trainset = [ex.with_inputs("prompt") for ex in examples]

    # ------------------------------------------------------------------ #
    # 2 .  Build (or wrap) the metric                                    #
    # ------------------------------------------------------------------ #
    if callable(llm_judge):
        translation_judge = llm_judge
    else:
        # Dynamically build a judge signature around the instruction string.
        judge_instructions = str(llm_judge).strip()

        class _AutoJudge(dspy.Signature):
            """{0}""".format(judge_instructions)
            english_sentence = dspy.InputField()
            french_translation = dspy.InputField()
            score: int = dspy.OutputField(desc="0 = bad, 1 = good")

        judge_predict = dspy.Predict(_AutoJudge)

        def translation_judge(example, prediction, trace=None):
            try:
                result = judge_predict(
                    english_sentence=example.prompt,
                    french_translation=prediction.get("generation", "")
                )
                return float(result.score)
            except Exception:
                return 0.0

    # ------------------------------------------------------------------ #
    # 3 .  Prepare the LM objects                                        #
    # ------------------------------------------------------------------ #
    def _to_lm(obj):
        if obj is None:
            return None
        return obj if isinstance(obj, dspy.LM) else dspy.LM(obj)

    teacher_lm = _to_lm(teacher_model)
    student_lm = _to_lm(student_model)

    # If the reader supplied no student, fall back to whatever DSPy is
    # already configured with; otherwise bind the student to our programme.
    if student_lm is not None:
        active_lm = student_lm
    else:
        active_lm = dspy.settings.get("lm")  # may still be None → DSPy default

    # ------------------------------------------------------------------ #
    # 4 .  Build the programme                                           #
    # ------------------------------------------------------------------ #
    class OptimSignature(dspy.Signature):
        """{0}""".format(system_prompt)

        prompt = dspy.InputField()
        generation = dspy.OutputField()

    programme = MyPredict(signature=OptimSignature)
    if active_lm is not None:
        programme.set_lm(active_lm)

    print(programme(prompt="I'm going to the convenience store."))
    print(programme.inspect_history())

    # # print(inspect_history())
    import pdb; pdb.set_trace()

    # ------------------------------------------------------------------ #
    # 5 .  Run MIPRO‑v2                                                  #
    # ------------------------------------------------------------------ #
    optimiser = dspy.MIPROv2(
        translation_judge,
        max_bootstrapped_demos=max_few_shots,
        max_labeled_demos=max_few_shots,
    )

    compile_kwargs = dict(
        trainset=trainset,
        requires_permission_to_run=False
    )
    if teacher_lm is not None:
        compile_kwargs["teacher"] = teacher_lm

    tuned_prog = optimiser.compile(programme, **compile_kwargs)

    # ------------------------------------------------------------------ #
    # 6 .  Extract the finished prompt (+ optional demos)                #
    # ------------------------------------------------------------------ #
    final_prompt = tuned_prog.signature.instructions.strip()

    if getattr(tuned_prog, "demos", None):
        final_prompt += "\n" + format_demos(tuned_prog.demos)

    import pdb; pdb.set_trace()

    return final_prompt


model = dspy.LM("gpt-4o-mini")
dspy.configure(lm = model)

optimized_system_prompt = optimize(
    training_inputs=[
        "I'm going to the convenience store.",
        "It's really cold out today."
    ],
    training_outputs=[
        "Je m'en vais au dépanneur.",
        "Il fait frette en maudit aujourd'hui."
    ],
    llm_judge="Return 1 if the French looks natural and French Canadian and 0 otherwise.",
    # llm_judge=translation_judge,
    # system_prompt = "Translate from english to french",
    system_prompt = "Please respond to the user's prompt.",
    max_few_shots = 0,
    teacher_model = None,
    student_model = model, 
)

print(optimized_system_prompt)
