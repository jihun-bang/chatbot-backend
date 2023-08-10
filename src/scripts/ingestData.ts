import {RecursiveCharacterTextSplitter} from 'langchain/text_splitter';
import {OpenAIEmbeddings} from 'langchain/embeddings/openai';
import {PDFLoader} from 'langchain/document_loaders/fs/pdf';
import {DirectoryLoader} from 'langchain/document_loaders/fs/directory';
import {FaissStore} from "langchain/vectorstores/faiss";

const filePath = 'docs';

export const run = async () => {
    try {
        console.log('split docs...');
        // PDF Load
        const directoryLoader = new DirectoryLoader(filePath, {
            '.pdf': (path) => new PDFLoader(path),
        });
        const rawDocs = await directoryLoader.load();

        // 텍스트 분할
        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200,
        });

        // PDF -> Docs
        const docs = await textSplitter.splitDocuments(rawDocs);
        console.log('creating vector store...');

        // 텍스트 임베딩 및 데이터 구조화
        const vectorStore = await FaissStore.fromDocuments(
            docs,
            new OpenAIEmbeddings()
        );
        await vectorStore.save('db');
    } catch (error) {
        console.log('error', error);
        throw new Error('Failed to ingest your data');
    }
};

(async () => {
    await run();
    console.log('ingestion complete');
})();
