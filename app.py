import base64
import html
import io
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(
    page_title="Generative AI Teacher",
    page_icon="🤖",
    layout="centered"
)

SUPPORTED_FILES = [
    "pdf", "txt", "docx", "doc",
    "csv", "xlsx", "xls",
    "jpg", "jpeg", "png",
    "mp4", "mov", "avi",
    "mp3", "wav", "m4a"
]

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "extra_attachments" not in st.session_state:
    st.session_state.extra_attachments = []
if "camera_key" not in st.session_state:
    st.session_state.camera_key = 0


@st.cache_resource
def create_chain():
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.3
    )

    prompt_template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a friendly AI teacher.
Explain concepts in simple English unless the user requests another language.
Use the entire conversation history to understand follow-up questions.
If file contents or image descriptions are provided in the prompt, use them clearly and helpfully."""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    return prompt_template | llm | StrOutputParser()


chain = create_chain()


def extract_file_context(files):
    """Extract readable text from lightweight text/csv files for the prompt."""
    snippets = []
    for f in files:
        name = f.name
        mime = f.type or ""
        if mime.startswith("image/"):
            b64_len = len(base64.b64encode(f.getvalue()).decode("utf-8"))
            snippets.append(f"[Image attached: {name}, {mime}, ~{b64_len} base64 chars]")
        elif name.lower().endswith((".txt", ".csv")):
            try:
                raw_text = f.getvalue().decode("utf-8", errors="ignore")
                trimmed = raw_text[:2500]
                snippets.append(f"[File: {name}]\n{trimmed}")
            except Exception:
                snippets.append(f"[File attached: {name}]")
        else:
            snippets.append(f"[File attached: {name} ({mime or 'binary'})]")
    return "\n\n".join(snippets)


def display_message(role, content, attachments=None):
    safe_content = html.escape(str(content)).replace("\n", "<br>")
    attach_html = ""
    if attachments:
        chips = "".join(
            f'<span style="display:inline-block; background:#EAF4FF; color:#123B63; '
            f'padding:2px 8px; border-radius:999px; font-size:12px; margin-top:6px; margin-right:6px;">'
            f'📎 {html.escape(a_name)}</span>'
            for a_name in attachments
        )
        attach_html = f"<div>{chips}</div>"

    if role == "user":
        chat_html = (
            '<div style="display:flex; justify-content:flex-end; '
            'align-items:flex-start; gap:10px; margin:14px 0;">'
            '<div style="background-color:#D9F0FF; color:#123B63; '
            'padding:13px 18px; border-radius:18px 4px 18px 18px; '
            'max-width:75%; overflow-wrap:anywhere;">'
            + safe_content + attach_html +
            '</div><div style="font-size:24px;">👤</div></div>'
        )
    else:
        chat_html = (
            '<div style="display:flex; justify-content:flex-start; '
            'align-items:flex-start; gap:10px; margin:14px 0;">'
            '<div style="font-size:24px;">✨</div>'
            '<div style="background-color:#EBF3FA; color:#102F4A; '
            'padding:13px 18px; border-radius:4px 18px 18px 18px; '
            'max-width:75%; overflow-wrap:anywhere;">'
            + safe_content +
            '</div></div>'
        )

    st.markdown(chat_html, unsafe_allow_html=True)


# Gemini-inspired clean styling
st.markdown("""
<style>
div[data-testid="stChatInput"],
div[data-testid="stChatInput"] > div,
div[data-testid="stChatInput"] textarea {
    background-color: #F0F4F9 !important;
    color: #1F1F1F !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
    color: #5F6368 !important;
}

div[data-testid="stChatInput"] {
    border: 1.5px solid #D3E3FD !important;
    border-radius: 24px !important;
}

button[data-testid="stPopoverButton"] {
    border-radius: 50% !important;
    width: 42px !important;
    height: 42px !important;
    padding: 0 !important;
    font-size: 20px !important;
    border: 1.5px solid #D3E3FD !important;
    background-color: #F0F4F9 !important;
    color: #1F1F1F !important;
}
</style>
""", unsafe_allow_html=True)

# Header
st.title("✨ Generative AI Teacher")
st.caption("Ask questions, attach files or photos, or snap with your camera.")

# Render chat history once
for msg in st.session_state.messages:
    display_message(msg["role"], msg["content"], msg.get("attachments"))

# Active extra attachments preview (from + menu: photos / files / camera)
if st.session_state.extra_attachments:
    st.markdown("**Attached via +:**")
    cols = st.columns(min(len(st.session_state.extra_attachments), 3))
    for idx, item in enumerate(st.session_state.extra_attachments):
        with cols[idx % len(cols)]:
            if item["kind"] == "image":
                st.image(item["bytes"], caption=item["name"], use_container_width=True)
            else:
                st.caption(f"📄 {item['name']}")
    if st.button("Clear + Attachments", key="clear_extra_att"):
        st.session_state.extra_attachments = []
        st.session_state.camera_key += 1
        st.rerun()

# Gemini-like bottom bar: [+] popover | chat input (with native attach) | clear chat
plus_col, input_col, clear_col = st.columns([0.09, 0.81, 0.10], vertical_alignment="bottom")

with plus_col:
    with st.popover("➕", help="Add photos, files, or use camera"):
        st.markdown("**Add Photos or Files**")
        picked_files = st.file_uploader(
            "Upload photos or documents",
            type=SUPPORTED_FILES,
            accept_multiple_files=True,
            key="plus_uploader"
        )
        if picked_files:
            existing_names = {x["name"] for x in st.session_state.extra_attachments}
            for pf in picked_files:
                if pf.name not in existing_names:
                    mime = pf.type or ""
                    st.session_state.extra_attachments.append({
                        "name": pf.name,
                        "type": mime,
                        "kind": "image" if mime.startswith("image/") else "file",
                        "bytes": pf.getvalue()
                    })
                    existing_names.add(pf.name)

        st.divider()
        st.markdown("**Camera**")
        cam_img = st.camera_input(
            "Take a photo",
            key=f"plus_camera_{st.session_state.camera_key}"
        )
        if cam_img is not None:
            cam_name = f"camera_{st.session_state.camera_key}.png"
            existing_names = {x["name"] for x in st.session_state.extra_attachments}
            if cam_name not in existing_names:
                st.session_state.extra_attachments.append({
                    "name": cam_name,
                    "type": "image/png",
                    "kind": "image",
                    "bytes": cam_img.getvalue()
                })

with input_col:
    submitted = st.chat_input(
        "Ask Gemini-style AI Teacher...",
        accept_file="multiple",
        file_type=SUPPORTED_FILES,
        key="gemini_chat_input"
    )

with clear_col:
    if st.button("🗑️", key="clear_chat_btn", help="Clear Chat"):
        st.session_state.messages = []
        st.session_state.extra_attachments = []
        st.rerun()

if submitted:
    user_text = (submitted.text or "").strip()
    inline_files = list(submitted.files or [])

    # Combine inline chat_input files + (+) popover files/camera photos
    combined_names = [f.name for f in inline_files] + [
        x["name"] for x in st.session_state.extra_attachments
    ]

    context_blocks = []
    for f in inline_files:
        context_blocks.append(extract_file_context([f]))
    for x in st.session_state.extra_attachments:
        if x["kind"] == "image":
            context_blocks.append(f"[Camera/Photo attached: {x['name']} ({x['type']})]")
        else:
            if x["name"].lower().endswith((".txt", ".csv")):
                txt = x["bytes"].decode("utf-8", errors="ignore")[:2500]
                context_blocks.append(f"[File: {x['name']}]\n{txt}")
            else:
                context_blocks.append(f"[File attached: {x['name']}]")

    attachments_summary = "\n\n".join(context_blocks)

    if not user_text and attachments_summary:
        question_for_llm = (
            f"The user attached the following items without extra text:\n{attachments_summary}\n\n"
            "Acknowledge what was attached and explain how you can help with it."
        )
        display_user_text = "Sent attachments"
    elif user_text and attachments_summary:
        question_for_llm = f"{user_text}\n\nAttached context:\n{attachments_summary}"
        display_user_text = user_text
    else:
        question_for_llm = user_text
        display_user_text = user_text

    if question_for_llm:
        st.session_state.messages.append({
            "role": "user",
            "content": display_user_text,
            "attachments": combined_names
        })

        chat_history = []
        for m in st.session_state.messages[:-1]:
            if m["role"] == "user":
                chat_history.append(HumanMessage(content=m["content"]))
            elif m["role"] == "assistant":
                chat_history.append(AIMessage(content=m["content"]))

        with st.spinner("Thinking..."):
            try:
                reply = chain.invoke({
                    "chat_history": chat_history,
                    "question": question_for_llm
                })
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply
                })
                # Clear extra attachments after sending so they don't repeat next turn
                st.session_state.extra_attachments = []
                st.session_state.camera_key += 1
                st.rerun()
            except Exception as e:
                st.session_state.messages.pop()
                st.error(f"Something went wrong: {e}")