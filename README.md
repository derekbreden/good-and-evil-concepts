# Good & Evil Words

100% browser based semantic embeddings demo.


### How it works

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