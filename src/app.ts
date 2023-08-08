import express, {Express, Request, Response} from 'express';
import * as serviceAccount from "./service_account_key.json" assert {type: 'json'};
import admin from "firebase-admin";
import {BufferMemory} from "langchain/memory";
import {FirestoreChatMessageHistory} from "langchain/stores/message/firestore";
import {ChatOpenAI} from "langchain/chat_models/openai";
import {ConversationalRetrievalQAChain} from "langchain/chains";
import dotenv from "dotenv"
import {FaissStore} from "langchain/vectorstores/faiss";
import {OpenAIEmbeddings} from "langchain/embeddings/openai";
import {PDFLoader} from "langchain/document_loaders/fs/pdf";
import {CharacterTextSplitter} from "langchain/text_splitter";
import {PromptTemplate} from "langchain/prompts";

const directory = "db";
const app: Express = express();
const port = 8080;
let chain: ConversationalRetrievalQAChain;

dotenv.config();
admin.initializeApp({
    credential: admin.credential.cert({
        projectId: serviceAccount.default.project_id,
        clientEmail: serviceAccount.default.client_email,
        privateKey: serviceAccount.default.private_key,
    }),
});

app.get('/', (_, res: Response) => {
    res.send('Typescript + Node.js + Express Server');
});

app.get('/chat', async (req: Request, res: Response) => {
    const question = req.query.question;
    console.log(`[question] ${question}`)
    const answer = await chain.call({question: question});
    console.log(`[answer] ${answer.text}`)
    res.send(answer);
});

app.listen(port, async () => {
    // const loader = new PDFLoader("src/manual/WPU-A1100C_230626.pdf");
    // const docs = await loader.load();
    // const splitter = new CharacterTextSplitter({
    //     chunkSize: 1000,
    //     chunkOverlap: 10,
    // });
    // const docOutput = await splitter.splitDocuments(docs);
    // const vectorStore = await FaissStore.fromDocuments(
    //     docOutput,
    //     new OpenAIEmbeddings()
    // );
    // await vectorStore.save(directory);
    const loadedVectorStore = await FaissStore.load(
        directory,
        new OpenAIEmbeddings()
    );
    const vectorStoreRetriever = loadedVectorStore.asRetriever({
            searchKwargs: {fetchK: 6}
        }
    );

    const memory = new BufferMemory(
        {
            memoryKey: "chat_history",
            chatHistory: new FirestoreChatMessageHistory({
                collectionName: "langchain",
                sessionId: "lc-example",
                userId: "a@example.com",
                config: {
                    projectId:
                        "skmagic-chatbot-develop",
                },
            }),
        });

    const model = new ChatOpenAI({
        modelName: "gpt-3.5-turbo-16k-0613",
        openAIApiKey: process.env.OPENAI_API_KEY,
        temperature: 0.7,
    });

    const promptTemplate = `
    You are '매직', an AI of SK매직 that solves customers' questions.
    Use the following pieces of context to answer the question at the end.
    If you have a question you don't know, don't make it up and direct it to the customer center at 1600-1661.
    Answer in Korean and with respect.
    If you have a question you don't know, don't make it up and direct it to the customer center at 1600-1661.
    Please respond in a soft, friendly tone, as if you were speaking to a customer service representative.
    Use emoticons if you need them.
    
    {context}
    
    Question: {question}
    Answer in Korean:`;
    const prompt = PromptTemplate.fromTemplate(promptTemplate)

    chain = ConversationalRetrievalQAChain.fromLLM(
        model,
        vectorStoreRetriever,
        {
            memory: memory,
            qaChainOptions: {
                type: "stuff",
                prompt: prompt,
            }
        }
    );
    console.log(`[server]: Server is running at http://localhost:${port}`);
});