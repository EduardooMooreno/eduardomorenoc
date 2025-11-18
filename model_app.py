"""model_app – Final Version (Hugging Face Evaluator + CSV export + Sentiment Analysis + Score Chart)"""

import os, json, random, textwrap, re, pandas as pd, torch
from pathlib import Path
from collections import Counter
import streamlit as st
import evaluate
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline as hf_pipeline,
)

# ===============================
# 1. DATA LOADING
# ===============================
@st.cache_data
def load_qa_data(path="Q&A_db_practice.json"):
    data_path = Path(path)
    if not data_path.exists():
        st.error(f"{path} not found in same folder as model_app.py.")
        return []
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)

qa_items = load_qa_data()
if not qa_items:
    st.stop()

# ===============================
# 2. AUTOMATIC EVALUATOR
# ===============================
rouge = evaluate.load("rouge")

def compute_rouge_l(ref, pred):
    return rouge.compute(predictions=[pred], references=[ref], use_stemmer=True)["rougeL"]

def extract_keywords(text, min_len=4, top_k=20):
    toks = [t.strip(".,;:!?()[]\"'").lower() for t in text.split() if len(t) >= min_len]
    return [w for w, _ in Counter(toks).most_common(top_k)]

def keyword_coverage(ref, pred, top_k=20):
    ref_kw = extract_keywords(ref, top_k=top_k)
    pred_toks = set(t.strip(".,;:!?()[]\"'").lower() for t in pred.split())
    present = [w for w in ref_kw if w in pred_toks]
    missing = [w for w in ref_kw if w not in pred_toks]
    return len(present)/max(1,len(ref_kw)), present, missing

def evaluate_answer(ref, ans):
    rouge_l = compute_rouge_l(ref, ans)
    cov, present, missing = keyword_coverage(ref, ans)
    score = round((0.5*rouge_l + 0.5*cov)*100,1)
    expl = [
        f"ROUGE-L: {rouge_l:.3f} | Keyword coverage: {cov:.3f}.",
        f"Included: {', '.join(present[:8])}." if present else "",
        f"Missing: {', '.join(missing[:8])}." if missing else "",
    ]
    return {"score":score,"explanation":" ".join(expl)}

# ===============================
# 3. LLM-BASED EVALUATOR
# ===============================
LLM_MODEL_NAME="Qwen/Qwen2.5-1.5B-Instruct"
LLM_SYSTEM_PROMPT=(
"You are a professor grading a student's short answer about a Machine Learning concept. "
"Compare the student's answer to the reference answer, explain briefly what is correct or missing, "
"and give a score from 0 to 100.\nRespond ONLY as:\nScore: <0-100>\nFeedback: <1-3 sentences>")

@st.cache_resource
def load_local_llm():
    tok=AutoTokenizer.from_pretrained(LLM_MODEL_NAME,trust_remote_code=True)
    mdl=AutoModelForCausalLM.from_pretrained(
        LLM_MODEL_NAME,device_map="auto",torch_dtype=torch.float32,trust_remote_code=True)
    mdl.eval();return tok,mdl
tokenizer,llm_model=load_local_llm()

@st.cache_resource
def load_sentiment_analyzer(): return hf_pipeline("sentiment-analysis")
sentiment_analyzer=load_sentiment_analyzer()

def call_llm_evaluator(q,ref,ans):
    msgs=[{"role":"system","content":LLM_SYSTEM_PROMPT},
           {"role":"user","content":f"Question:{q}\n\nReference:{ref}\n\nStudent:{ans}"}]
    prompt=tokenizer.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    with torch.no_grad():
        inp=tokenizer(prompt,return_tensors="pt"); inp={k:v.to(llm_model.device) for k,v in inp.items()}
        out_ids=llm_model.generate(**inp,max_new_tokens=220,temperature=0.4,top_p=0.9,
                                   eos_token_id=tokenizer.eos_token_id)
        text=tokenizer.decode(out_ids[0][inp["input_ids"].shape[1]:],skip_special_tokens=True).strip()
    st.sidebar.text_area("LLM raw output (debug)",text,height=140)
    score=75.0; fb=text
    m1=re.search(r"Score\s*[:\-]\s*(\d{1,3})",text,re.I)
    m2=re.search(r"(Feedback|Explanation)\s*[:\-]\s*(.*)",text,re.I|re.S)
    if m1:
        try: score=float(m1.group(1))
        except: pass
    if m2: fb=m2.group(2).strip()
    return {"score":max(0,min(100,score)),"analysis":fb}

# ===============================
# 4. STATE
# ===============================
def sample_question(lst): return random.choice(lst)
if "current_qa" not in st.session_state: st.session_state.current_qa=None
if "history" not in st.session_state: st.session_state.history=[]

# ===============================
# 5. STREAMLIT UI
# ===============================
st.set_page_config(page_title="ML Q&A Evaluator",page_icon="🤖",layout="wide")
st.title("🤖 ML Concept Q&A Evaluator")
st.write("Ask, answer, and evaluate ML concepts with automatic metrics, an LLM grader, "
         "sentiment analysis of your feedback, CSV export, and score charts.")

with st.sidebar:
    st.header("Controls")
    if st.button("🔄 New random question"):
        st.session_state.current_qa=sample_question(qa_items)
        for k in ["last_answer","last_eval_auto","last_eval_llm"]: st.session_state.pop(k,None)
    if st.button("🧼 Reset session"):
        st.session_state.current_qa=None; st.session_state.history=[]
        for k in ["last_answer","last_eval_auto","last_eval_llm"]: st.session_state.pop(k,None)

    st.markdown("---")
    st.subheader("Session stats")
    st.write(f"Questions answered: **{len(st.session_state.history)}**")
    if st.session_state.history:
        avg=sum(h["evaluation"]["automatic"]["score"] for h in st.session_state.history)/len(st.session_state.history)
        st.write(f"Average automatic score: **{avg:.1f}/100**")

        # 📊 SCORE CHART
        df=pd.DataFrame({
            "Automatic":[h["evaluation"]["automatic"]["score"] for h in st.session_state.history],
            "LLM":[h["evaluation"]["llm"]["score"] for h in st.session_state.history]
        })
        st.subheader("📈 Score Trend")
        st.line_chart(df,use_container_width=True)

    # 💾 CSV EXPORT
    st.markdown("---"); st.subheader("💾 Export Results")
    if st.session_state.history:
        export=[]
        for h in st.session_state.history:
            export.append({
                "Question":h["question"],
                "Reference":h["reference_answer"],
                "Student":h["student_answer"],
                "Auto Score":h["evaluation"]["automatic"]["score"],
                "Auto Explanation":h["evaluation"]["automatic"]["explanation"],
                "LLM Score":h["evaluation"]["llm"]["score"],
                "LLM Feedback":h["evaluation"]["llm"]["analysis"],
                "User Feedback":h.get("user_feedback",""),
                "Feedback Sentiment":h.get("feedback_sentiment",""),
                "Feedback Confidence":h.get("feedback_confidence","")
            })
        csv=pd.DataFrame(export).to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV",data=csv,file_name="llm_eval_results.csv",mime="text/csv")
    else: st.caption("Answer at least one question to enable export.")

    st.markdown("---"); st.caption("Assignment 11.00 – LLM Evaluator (Streamlit UI)")

# ===============================
# MAIN
# ===============================
if st.session_state.current_qa is None: st.session_state.current_qa=sample_question(qa_items)
qa=st.session_state.current_qa; q,ref=qa["question"],qa["answer"]
st.subheader("Current Question"); st.write(q)
ans=st.text_area("Type your answer here:",value=st.session_state.get("last_answer",""),
                 height=160,placeholder="Write your explanation...")
col1,col2=st.columns([1,2])
with col1: submit=st.button("✅ Submit answer")
with col2: show_ref=st.checkbox("Show reference answer after evaluation")

if submit and ans.strip():
    st.session_state.last_answer=ans
    eval_auto=evaluate_answer(ref,ans)
    eval_llm=call_llm_evaluator(q,ref,ans)
    st.session_state.last_eval_auto,st.session_state.last_eval_llm=eval_auto,eval_llm
    st.session_state.history.append({
        "question":q,"reference_answer":ref,"student_answer":ans,
        "evaluation":{"automatic":eval_auto,"llm":eval_llm},
    })

if "last_eval_auto" in st.session_state:
    ea,el=st.session_state.last_eval_auto,st.session_state.last_eval_llm
    st.markdown("## Evaluation Results")
    st.markdown("### 🔍 Automatic Evaluation"); st.write(f"**Score:** `{ea['score']}`"); st.write(textwrap.fill(ea["explanation"],100))
    st.markdown("### 🧠 LLM Evaluation (Qwen2.5 Instruct)"); st.write(f"**Score:** `{el['score']}`"); st.write(textwrap.fill(el["analysis"],100))
    if show_ref: st.markdown("### 📘 Reference Answer"); st.write(textwrap.fill(ref,100))

    # 💬 FEEDBACK + SENTIMENT
    st.markdown("### 💬 Your Feedback")
    feedback=st.text_area("What do you think of this evaluation?",key=f"fb_{len(st.session_state.history)}",height=100)
    if st.button("Submit Feedback"):
        if feedback.strip():
            res=sentiment_analyzer(feedback)[0]; lbl=res["label"]; sc=round(res["score"],3)
            st.success(f"Sentiment: **{lbl}** ({sc})")
            if st.session_state.history:
                st.session_state.history[-1].update({
                    "user_feedback":feedback,
                    "feedback_sentiment":lbl,
                    "feedback_confidence":sc})
        else: st.warning("Please write a short comment first.")
