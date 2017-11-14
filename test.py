import tensorflow as tf
import model.knowledge_base as kb

L = kb.KnowledgeBase([300,300], 50)
s = kb.PathPolicy(10,5, L)