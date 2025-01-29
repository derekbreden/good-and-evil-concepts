# Polarized Words

https://defenderofbasic.github.io/good-and-evil-concepts/

100% browser based semantic embeddings demo. Computes the embedding of the input words and shows their relative distance to the pole words ("good" and "evil"). 

Here is Obama, Trump, Hitler, Kittens. Hitler is "evil" as expected. Trump is more "evil" than Obama, but both of them are way better than Hitler. Kittens is not as "good" as Obama. 

<img src="https://github.com/user-attachments/assets/1812db4f-0061-4f5a-a304-65bc44048fb2" width=400></img>

### What do these distances mean?

Concretely, these are word associations based on the training data of this language model. If the word "trump" appears with negative words a lot in newspapers, social media, etc, it will be closer to those words. 

**What it effectively tells us is how our culture talks about these ideas**. It will reflect whatever biases or connections are in the training data. It's a way to poke inside the mind of a language model, and thus, the collective consciousness of society. 

If we try polarizing words like "capitalism" and "communism", you'll get "communism" being slightly closer to "good" than capitalism. What this tells us is that, _in aggregate_, there's more net positive discourse around communism, and perhaps more talk of the evils of capitalism. 

<img src="https://github.com/user-attachments/assets/49008494-2af3-4d9d-b125-e172a7ed0f7b" width=400></img>

But society is not a monolith. **The aggregate view doesn't let us discern** between (1) everyone feels this specific way or (2) different tribes vehemently disagree. If we were to ask people to rate these words, we might see a bimodal distribution where many put "capitalism" as very good and communism as very evil, and others the reverse. 

We can validate our findings by surveying people, and by searching for corrborating evidence in the embedding relationships. We can confirm that "capitalism" leans more right wing, and "communism" leans more left wing. And to sanity check, "Obama" and "trucks" fall in the positions we expect along this line:

<img src="https://github.com/user-attachments/assets/6bbabde4-eab0-4f5e-8511-dbbaa8f64297" width=400></img>


### How the code works

Can get any model from here: https://huggingface.co/models?other=feature-extraction


```javascript
import { pipeline, env } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.6.0';
env.allowLocalModels = false;
const model_name = 'nomic-ai/nomic-embed-text-v1.5'

const embedder = await pipeline('feature-extraction', model_name,
{
    quantized: true,
    progress_callback: data => {
        const { progress, loaded, total } = data
        if (progress) {
            const totalMB = Math.round(total / (1024 * 1024))
            const loadedMB = Math.round(loaded / (1024 * 1024))
            
            console.log(`${Math.round(progress)}% (${loadedMB}/${totalMB} mb)`);
        }
    }
});

const inputText = 'Hello world!'
const embeddingVector = (await embedder(inputText, {pooling: 'mean', normalize: true})).data
```
