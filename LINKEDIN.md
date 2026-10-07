# LinkedIn launch kit for AARUL

## The post

> I built my own AI language model from scratch, and trained it on my laptop.
>
> Not a ChatGPT wrapper. Not a fine-tuned open-source model. I wrote every piece myself:
>
> → the tokenizer that turns text into numbers
> → the transformer (same design as Meta's Llama, 15.7M parameters)
> → the training loop: 2.6 hours on my RTX 3070 Ti laptop
> → the engine that runs it in your browser, in 19 KB of TypeScript
>
> Cloud bill: $0.
>
> The part I can't stop playing with: you can watch it think. Point at any word it writes and you see the other words it considered, with their probabilities, and which earlier words its 48 attention heads were looking at when it chose.
>
> You can also watch it learn. Same prompt, different points in training:
>
> Step 0: "Once upon a timeJohn reading distant distant belt elephant…"
> Step 50: "Once upon a time, the big, a big, and, you, mom."
> Step 7,000: "One day, Tim found a key in his room. He did not know what it was for, but he thought it was pretty. Tim showed the key to his mom. 'Mom, look what I found!'"
>
> 3 things I learned:
>
> 1. LLMs aren't magic. The model is ~150 lines of code. The magic is data and compute.
> 2. A laptop is not a training rig. My GPU hit 87°C within minutes and throttled itself to less than half speed. I had to measure that and size the whole run around it.
> 3. "It works" isn't the same as "it's right." I implemented the model twice, once in PyTorch for training and once in TypeScript for the browser, and a test checks they agree to 7 decimal places. A small bug there would still produce plausible text, just worse, and I'd never notice.
>
> It has only ever read children's stories, so it has some opinions. I typed "The stock market" and it wrote: "The stock market was a very big store." 😄
>
> Try it (it runs entirely in your browser): https://aarulkoul.github.io/aarul/
> Code: https://github.com/AarulKoul/aarul
>
> What should I train it on next? 👇
>
> #MachineLearning #AI #LLM #BuildInPublic #DeepLearning

## What to attach

LinkedIn favours native video over links, so attach a **20–30 second screen recording**:

1. (0–8s) Click a prompt chip in "watch it think" and let it write in **watch** speed, with the
   inspector panel following each word.
2. (8–15s) Point at a surprising (purple) word: the alternatives and the amber attention glow show up.
3. (15–25s) Scroll to "watch it learn" and press **Play**: gibberish turns into a story.

Windows: press `Win + Alt + R` (Xbox Game Bar) to record, or use Snipping Tool's video mode.

## Tips

- Many people put the link in the first comment instead of the post body. Either works. Just make
  sure the demo link is somewhere people can tap.
- The question at the end ("what should I train it on next?") is what drives comments. Reply to
  every comment in the first hour.
- Pin a comment with a **share link** to a story it wrote that made you laugh. Every story has a
  "Share this story" button, and the link replays the exact same story for whoever opens it.
- Post Tuesday to Thursday, mid-morning in your timezone.
