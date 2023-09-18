import initialize from './init.js';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { PineconeClient } from '@pinecone-database/pinecone'
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { RetrievalQAChain } from 'langchain/chains';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { PromptTemplate } from 'langchain/prompts';
import EmbeddingModel from './embedding-model.js';
import { loadTextDocument, writeObjectToFile } from './util.js';
import ChatModel from './chat-model.js';
import PromptSync from 'prompt-sync';

initialize();

const embeddingModel = new EmbeddingModel();

console.log('connecting to pinecone')
const client = new PineconeClient();
await client.init({
    apiKey: process.env.PINECONE_API_KEY,
    environment: process.env.PINECONE_ENVIRONMENT,
});

const pineconeIndex = client.Index(process.env.PINECONE_INDEX);

console.log('upserting documents')

const files = [
    // 'bits-acceptable-use-policy.txt',
    // 'community-guidelines.txt',
    // 'community-guidelines-faq.txt',
    // 'dmca-guidelines.txt',
    // 'list-of-prohibited-games.txt',
    // 'terms-of-sale.txt',
    // 'terms-of-service.txt',
    // 'trademark-guidelines.txt',
];

files.forEach(async (file) => {
    console.log(`loading document:  + ${file}`);
    const documents = await loadTextDocument(`documents/twitch/${file}`);

    console.log('splitting document...');
    const splitter = new RecursiveCharacterTextSplitter({
        chunkSize: 2000,
        chunkOverlap: 100,
    });

    const splitDocuments = await splitter.splitDocuments(documents);

    try {
        await PineconeStore.fromDocuments(splitDocuments, embeddingModel.model, {
            pineconeIndex
        });
    } catch (error) {
        console.log(error);
    }
    console.log('done\n\n');
});

const vectorStore = await PineconeStore.fromExistingIndex(
    new OpenAIEmbeddings(),
    { pineconeIndex }
);

const chatModel = new ChatModel('gpt-4');

const template = `
    Use the following pieces of context to answer the question at the end.
    You are an AI assistant to help twitch streamers know what activities and content falls within Twitch's terms of service and community guidelines.
    Don't say to review the guidelines or terms of service, it's your job to give them guidance.
    Do not say you are an AI assistant.
    If you don't know the answer, say you don't know. Do not make up an answer.
    Only answer questions that pertain to activity on Twitch.

    Question: {question}`;

const chain = RetrievalQAChain.fromLLM(chatModel.model, vectorStore.asRetriever(), {
    prompt: PromptTemplate.fromTemplate(template),
    returnSourceDocuments: true
});

const prompt = PromptSync();

while (true) {

    const query = prompt('Ask a question (type exit to exit): ');

    if (query === 'exit') {
        break;
    }

    const response = await chain.call({ query });
    console.log('\n\nQuestion:\n\n' + query + '\n\nAnswer:\n\n' + response.text + '\n\n')

    // write response to file with the current tiemstamp in name
    const date = new Date();
    const timestamp = date.getTime();
    await writeObjectToFile(response, `response-${timestamp}.json`);
}

