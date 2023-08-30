import express from 'express';
import * as serviceAccount from "./service_account_key.json" assert {type: 'json'};
import admin from "firebase-admin";
import {ConversationalRetrievalQAChain} from "langchain/chains";
import dotenv from "dotenv"
import {FaissStore} from "langchain/vectorstores/faiss";
import {OpenAIEmbeddings} from "langchain/embeddings/openai";
import {makeAgentChain, makeChain} from "./chain.js";

let loadedVectorStore: FaissStore;
const directory = "db";
const port = 8080;

const app = express();
app.use(express.json());
app.use(express.urlencoded());

dotenv.config();
admin.initializeApp({
    credential: admin.credential.cert({
        projectId: serviceAccount.default.project_id,
        clientEmail: serviceAccount.default.client_email,
        privateKey: serviceAccount.default.private_key,
    }),
});

app.get('/', (_, res) => {
    res.send('Typescript + Node.js + Express Server');
});

app.post('/api/chat', async (req, res) => {
    try {
        const {question, session_id, user_id} = req.body;
        if (!question) {
            res.status(400).json({message: '질문 해주세요!'});
            return
        } else if (!session_id || !user_id) {
            res.status(400).json({message: '잘못된 형식으로 요청하였습니다.'});
            return
        }
        const sanitizedQuestion = question.trim().replaceAll('\n', ' ');
        console.log(`[question] ${question}`)
        const chain = await makeAgentChain(loadedVectorStore, session_id, user_id);
        const response = await chain.call({question: sanitizedQuestion});
        console.log('[response]', response);
        res.status(200).json(response);
    } catch (e) {
        res.status(400).json({
            message: '잘못된 형식으로 요청하였습니다.',
            error: `${e}`
        });
    }
});

app.post('/api/agent', async (req, res) => {
    try {
        const {question, session_id, user_id} = req.body;
        if (!question) {
            res.status(400).json({message: '질문 해주세요!'});
            return
        } else if (!session_id || !user_id) {
            res.status(400).json({message: '잘못된 형식으로 요청하였습니다.'});
            return
        }
        const sanitizedQuestion = question.trim().replaceAll('\n', ' ');
        console.log(`[question] ${question}`)
        const chain = await makeChain(loadedVectorStore, session_id, user_id);
        const response = await chain.call({question: sanitizedQuestion});
        console.log('[response]', response);
        res.status(200).json(response);
    } catch (e) {
        res.status(400).json({
            message: '잘못된 형식으로 요청하였습니다.',
            error: `${e}`
        });
    }
});


app.listen(port, async () => {
    loadedVectorStore = await FaissStore.load(
        directory,
        new OpenAIEmbeddings()
    );
    console.log(`[server]: Server is running at http://localhost:${port}`);
});