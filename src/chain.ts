import {FaissStore} from 'langchain/vectorstores/faiss';
import {ConversationalRetrievalQAChain} from 'langchain/chains';
import {ConversationSummaryMemory} from "langchain/memory";
import {FirestoreChatMessageHistory} from "langchain/stores/message/firestore";
import {ChatOpenAI} from "langchain/chat_models";

const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `
You are the AI of SK매직 and your name is 매직.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
Use emoticons if you need them.

{context}

Question: {question}
Helpful answer in Korean:`;

export const makeChain = async (vectorStore: FaissStore) => {
    // 모델 생성
    const model = new ChatOpenAI({
        temperature: 0.6,
        modelName: 'gpt-3.5-turbo-0613',
    });

    // Firebase Store 기록 및 내역 조회
    const firestoreHistory = new FirestoreChatMessageHistory({
        collectionName: "langchain",
        sessionId: "lc-example",
        userId: "a@example.com",
        config: {
            projectId:
                "skmagic-chatbot-develop",
        },
    });

    /// 채팅 내역 요약 및 메모리 처리
    const bufferMemory = new ConversationSummaryMemory(
        {
            memoryKey: "chat_history",
            llm: model,
            chatHistory: firestoreHistory,
        });

    /// Chain 생성
    return ConversationalRetrievalQAChain.fromLLM(
        model,
        vectorStore.asRetriever(),
        {
            memory: bufferMemory,
            qaTemplate: QA_PROMPT,
            questionGeneratorTemplate: CONDENSE_PROMPT,
            returnSourceDocuments: false,
            verbose: true,
        },
    );
};
