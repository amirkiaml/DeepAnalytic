"""
app.py

Streamlit pilot interface for DeepAnalytic. Wraps rag_pipeline.py's
query_multi_concat() -- the non-reranked path, chosen because three
separate tests across this project found reranking flat-to-harmful.

Deliberately minimal: one pipeline, no mode switching, no settings for
testers to fiddle with. The point of the pilot is to find out whether
answers are useful to philosophers, not to have them evaluate the
architecture.

Deploy notes:
  - Secrets go in Streamlit's secrets manager, not a .env file.
  - Streamlit Community Cloud has an ephemeral filesystem, so logs are
    written to Google Sheets rather than to disk.
"""

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone

import streamlit as st

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

# Resolved relative to this file rather than the working directory, since a
# bare "assets/logo.png" only works when Streamlit is launched from the
# project root. Committed to the repo, not gitignored: Streamlit Cloud only
# has what's in git, so a local-only asset would work in development and
# silently fall back to the emoji on deploy.
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "logo.png")

MAX_QUERIES_PER_SESSION = 15
MEMORY_TURNS = 3          # how many previous exchanges to pass as context
K_PER_SUBQUERY = 3
MAX_SUBQUERIES = 5

# Chunks shorter than this are hidden from the source display. About 1.4% of
# the corpus is under 100 characters, almost all of it section headings that
# landed alone at a chunk boundary ("3. Space, Body, and Motion" and the like).
# They carry no information and look like a malfunction when cited as a source.
# Filtered at display only: the model still receives them in context, where
# they appear to be harmlessly ignored. The proper fix is a length filter at
# ingest time, but that means re-indexing 111,048 chunks for a 1.4% problem.
MIN_SOURCE_CHARS = 100

# Vector search always returns its nearest neighbours, however poor the match.
# There's no similarity floor below which it returns nothing, so a question the
# corpus can't answer still comes back with k results -- they're just bad ones.
# When the model correctly says it found nothing, displaying those results
# anyway makes the interface look like it retrieved something useful and
# ignored it.
#
# This is a stopgap. The principled fix is a similarity threshold, which needs
# scores surfaced through the pipeline (similarity_search_with_score rather
# than similarity_search). Matching on refusal phrasing is brittle: it depends
# on wording the generation prompt encourages but doesn't guarantee, so it will
# miss refusals phrased differently.
# Note on what's deliberately absent: "context does not contain" was here
# initially and had to be removed. It matched not only outright refusals but
# also partial answers of the form "the context does not contain the answer to
# X, but it does discuss Y" -- which have real sources worth showing. The
# markers below all assert a total absence of information, which is the only
# case where hiding sources is right.
# Plural forms matter here. The generation prompt was changed to say
# "passages" rather than "context", and refusals immediately started coming
# back as "The passages provided do not contain any information about..." --
# "do not" rather than "does not" -- which silently stopped matching and let
# irrelevant sources display again. Worth remembering that changing the
# generation prompt can break this matcher without any error surfacing.
REFUSAL_MARKERS = [
    "does not contain any information",
    "do not contain any information",
    "doesn't contain any information",
    "don't contain any information",
    "does not provide information",
    "do not provide information",
    "doesn't provide information",
    "don't provide information",
    "no information about",
    # More direct than any phrasing about what the passages contain, and the
    # model appends it to most refusals under the current prompt.
    "cannot answer the question based on",
    "can't answer the question based on",
]

# Answers longer than this are treated as substantive even if they contain
# refusal phrasing somewhere, on the reasoning that a genuine refusal is
# brief. See looks_like_refusal().
#
# Started at 400 and had to be raised. The better refusals aren't bare
# "no" answers -- they explain what the corpus does contain and why it
# doesn't settle the question, which is more useful to a reader but runs
# past 400 characters easily. One such refusal landed at 396, right at the
# edge, which made clear the threshold was cutting through a category
# rather than between two.
REFUSAL_MAX_CHARS = 600

st.set_page_config(
    page_title="SEP Assistant",
    # Falls back to an emoji if the file is missing, so a fresh checkout
    # without the asset still runs rather than erroring on startup.
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else "📚",
    layout="centered",
)

# Resolved once rather than per message. st.chat_message accepts a path or
# an emoji; passing a missing path raises, so this guards against it.
ASSISTANT_AVATAR = LOGO_PATH if os.path.exists(LOGO_PATH) else "📚"


# ---------------------------------------------------------------------
# Logging (Google Sheets, toggleable)
# ---------------------------------------------------------------------

def logging_enabled() -> bool:
    """Logging is on only if explicitly enabled in secrets and credentials exist."""
    return bool(st.secrets.get("LOGGING_ENABLED", False)) and "gcp_service_account" in st.secrets


@st.cache_resource
def get_worksheets():
    """
    Returns (interactions, feedback, chunks) worksheets, or (None, None,
    None) if logging is off or misconfigured. Cached so we don't
    re-authenticate on every rerun.

    Three tabs rather than one, for two different reasons.

    Feedback is separate because it arrives after the interaction row is
    already written. Updating that row in place would mean tracking its row
    number, which is fragile if anything else writes concurrently. A
    separate tab keyed by interaction_id is simpler and joins cleanly.

    Chunk text is separate because it's bulky. Five passages at several
    hundred characters each would make the interactions tab unreadable in
    the browser, and that tab is the one worth being able to skim. Full
    passages go here for later analysis, one row per passage, joined back
    on interaction_id.
    """
    if not logging_enabled():
        return None, None, None
    try:
        import gspread
        from google.oauth2.service_account import Credentials

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        creds = Credentials.from_service_account_info(
            dict(st.secrets["gcp_service_account"]), scopes=scopes
        )
        client = gspread.authorize(creds)
        book = client.open_by_key(st.secrets["SHEET_ID"])

        interactions = book.sheet1

        # Create the feedback tab on first use rather than requiring manual
        # setup, so a fresh deployment works without a preparation step.
        try:
            feedback = book.worksheet("feedback")
        except Exception:
            feedback = book.add_worksheet(title="feedback", rows=1000, cols=7)
            feedback.append_row([
                "timestamp", "user_id", "interaction_id",
                "answer_rating", "source_rating", "comment", "question",
            ])

        try:
            chunks = book.worksheet("chunks")
        except Exception:
            chunks = book.add_worksheet(title="chunks", rows=5000, cols=8)
            chunks.append_row([
                "timestamp", "user_id", "interaction_id", "position",
                "title", "section", "url", "text",
            ])

        return interactions, feedback, chunks
    except Exception as e:
        # Logging failing should never break the app for a tester.
        print(f"Logging setup failed: {e}")
        return None, None, None


def anonymous_id(email: str) -> str:
    """
    Stable pseudonymous identifier. Lets us group one tester's questions
    together for analysis without storing their email alongside their
    interactions.
    """
    salt = st.secrets.get("ID_SALT", "deepanalytic")
    return hashlib.sha256(f"{salt}:{email.lower().strip()}".encode()).hexdigest()[:12]


def log_interaction(user_id: str, interaction_id: str, question: str,
                    answer: str, sources: list, subqueries: list):
    ws, _, chunk_ws = get_worksheets()
    if ws is None:
        return
    try:
        source_summary = " | ".join(
            f"{s.get('title')} > {s.get('section')}" for s in sources
        )
        now = datetime.now(timezone.utc).isoformat()
        ws.append_row([
            now, user_id, interaction_id, question, answer,
            source_summary, json.dumps(subqueries),
        ])

        # Full passage text goes to its own tab, one row per passage.
        # append_rows rather than a loop: one API call instead of five,
        # which matters against Sheets' write quota.
        if chunk_ws is not None and sources:
            chunk_ws.append_rows([
                [
                    now, user_id, interaction_id, i,
                    s.get("title", ""), s.get("section", ""),
                    s.get("source", ""), s.get("text", ""),
                ]
                for i, s in enumerate(sources, 1)
            ])
    except Exception as e:
        print(f"Logging failed: {e}")


def log_feedback(user_id: str, interaction_id: str, answer_rating: str,
                 source_rating: str, comment: str, question: str):
    _, ws, _ = get_worksheets()
    if ws is None:
        return
    try:
        ws.append_row([
            datetime.now(timezone.utc).isoformat(),
            user_id,
            interaction_id,
            answer_rating,
            source_rating,
            comment,
            question,
        ])
    except Exception as e:
        print(f"Feedback logging failed: {e}")


# ---------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------

def check_login(email: str, password: str) -> bool:
    """
    Each tester gets their own password, stored in secrets as a
    [passwords] table mapping email -> password. One password per person
    so access can be revoked individually.
    """
    users = st.secrets.get("passwords", {})
    expected = users.get(email.lower().strip())
    return expected is not None and password == expected


def login_screen():
    st.title("SEP Assistant")
    st.caption("A pilot. Access is by invitation.")

    with st.form("login"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Sign in")

    if submitted:
        if check_login(email, password):
            st.session_state.authenticated = True
            st.session_state.user_id = anonymous_id(email)
            st.rerun()
        else:
            st.error("Email or password not recognised.")


# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------

@st.cache_resource
def get_rag():
    """One pipeline instance per app process, not per user session."""
    from rag_pipeline import RerankRAG
    return RerankRAG()


def build_generation_prompt(question: str, history: list) -> str:
    """
    Builds what the model is asked to answer, with recent conversation
    prepended so a follow-up like "how does that relate to physicalism"
    can resolve what "that" refers to.

    This is deliberately kept separate from what gets *searched*. An
    earlier version passed this same string to retrieval, which polluted
    the search badly: asking about the US president after asking about
    the meaning of life retrieved passages about the meaning of life,
    because the prior topic was sitting in the query text competing for
    matches. Retrieval now sees only the current question.

    rag_pipeline.py remains stateless. All conversation memory lives
    here, in the app layer.
    """
    if not history:
        return question

    recent = history[-MEMORY_TURNS:]
    lines = []
    for turn in recent:
        lines.append(f"Earlier question: {turn['question']}")
        lines.append(f"Earlier answer (abridged): {turn['answer'][:300]}")

    context = "\n".join(lines)
    return (
        f"Earlier in this conversation:\n{context}\n\n"
        f"Answer this question, resolving any references it makes to the "
        f"exchange above: {question}"
    )


# ---------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------

def main_app():
    st.title("SEP Assistant")

    with st.expander("What this is, and what to expect", expanded=False):
        st.markdown(
            """
            This answers questions using **only** the Stanford Encyclopedia of
            Philosophy. It searches about 1,800 entries and builds an answer from
            the passages it finds, so what you get should reflect what the entries
            actually say rather than a model's general impressions.

            **A few things worth knowing.**

            It handles multi-part questions by splitting them up behind the scenes,
            searching for each part separately, then combining what it finds. That
            works better than you might expect on questions that bundle two or three
            things together. Some examples of the kind of thing it copes with:

            - *How does Anselm reconcile God's mercy with God's justice, and how does
              that connect to his account of freedom and sin?*
            - *What's the difference between induction and abduction, and what is van
              Fraassen's argument of the bad lot?*
            - *How does Abelard's rejection of universals relate to his view that
              intentions rather than deeds determine moral worth?*
            - *Compare the disinterest thesis in aesthetics with Danto's argument
              about Brillo Boxes.*

            It has a short memory, so a follow-up like "how does that relate to
            physicalism" will usually work. But please don't lean on it.
            **Self-contained questions retrieve better, and they're also the unit the
            next version of this tool is being built around, since checking a paper
            means checking individual claims rather than following a conversation.**

            It's an early pilot and it will have rough edges. Finding where it falls
            over is more useful to me than finding where it works.

            Sources are shown under each answer. I'd genuinely value your view on
            whether they're the right ones, not just whether the answer reads well.
            There's a feedback box under each answer for exactly that.

            Questions, answers, and any feedback you leave are logged anonymously so
            I can see what's working. Your email isn't stored alongside them.
            """
        )

    if "history" not in st.session_state:
        st.session_state.history = []

    used = len(st.session_state.history)
    remaining = MAX_QUERIES_PER_SESSION - used

    if remaining <= 3:
        st.caption(f"{remaining} questions left this session.")

    # Replay the conversation so far
    for turn_index, turn in enumerate(st.session_state.history):
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
            st.write(turn["answer"])
            if looks_like_refusal(turn["answer"]):
                st.caption(
                    "Nothing in the encyclopedia matched this closely enough to cite."
                )
            else:
                render_sources(turn["sources"], turn["subqueries"], turn.get("origins"))
            render_feedback(turn_index, turn)

    if remaining <= 0:
        st.warning(
            "You've reached the limit for this session. Refresh the page to start "
            "a new one, or get in touch if you want to keep going."
        )
        return

    question = st.chat_input("Ask a question about philosophy")
    if not question:
        return

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("Searching the encyclopedia..."):
            try:
                rag = get_rag()
                result = rag.query_multi_concat(
                    question,
                    max_subqueries=MAX_SUBQUERIES,
                    k_per_subquery=K_PER_SUBQUERY,
                    generation_question=build_generation_prompt(
                        question, st.session_state.history
                    ),
                )
            except Exception as e:
                st.error(
                    "Something went wrong answering that. If this keeps happening, "
                    "let me know what you asked."
                )
                print(f"Query failed: {e}")
                return

        answer = result["answer"]
        sources = result.get("sources", [])
        subqueries = result.get("subqueries", [])
        origins = result.get("source_origins", [])

        st.write(answer)
        if looks_like_refusal(answer):
            st.caption(
                "Nothing in the encyclopedia matched this closely enough to cite."
            )
        else:
            render_sources(sources, subqueries, origins)

    interaction_id = str(uuid.uuid4())[:8]

    st.session_state.history.append({
        "question": question,
        "answer": answer,
        "sources": sources,
        "subqueries": subqueries,
        "origins": origins,
        "interaction_id": interaction_id,
    })

    log_interaction(
        st.session_state.user_id, interaction_id, question, answer, sources, subqueries
    )

    # Rerun so the exchange is drawn by the replay loop above, which is
    # where the feedback widget lives. Without this the newest answer is
    # the one turn without a feedback box until the next question.
    st.rerun()


def render_feedback(turn_index: int, turn: dict):
    """
    Feedback controls under one answer.

    Per-answer rather than one form at the end of a session, because the
    useful question is which *specific* answers were good and why. A
    session-level form loses that link, and by the end people have
    forgotten which answer they meant.

    Given a visible border and a coloured heading because in the first
    version it looked identical to everything else on the page and got
    overlooked. Testers giving up their time won't hunt for the feedback
    box; it has to be obvious without shouting.

    Two ratings rather than one. An answer can read well while citing the
    wrong passages, or cite exactly the right passages and still miss the
    point, and those failures need different fixes. Collapsing them into a
    single score would hide which one happened.

    Feedback is written to the sheet on submit rather than held until
    session end, so a tester who closes the tab doesn't lose it.
    """
    if turn.get("feedback_submitted"):
        st.markdown(
            "<div style='background:#e8f5e9; border-left:4px solid #43a047; "
            "padding:0.6em 1em; border-radius:4px; margin:0.5em 0; "
            "font-size:0.9em; color:#2e7d32'>Thanks, that's logged.</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        "<div style='background:#fff8e1; border-left:4px solid #ffa726; "
        "padding:0.7em 1em; border-radius:4px; margin:0.8em 0 0.2em 0'>"
        "<span style='font-weight:600; color:#e65100'>How was this answer?</span>"
        "<br><span style='font-size:0.85em; color:#6d4c41'>"
        "This is the bit I actually learn from. Even one line helps."
        "</span></div>",
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            answer_rating = st.radio(
                "**The answer**",
                ["Good", "Partly right", "Wrong or unhelpful"],
                key=f"answer_rating_{turn_index}",
                index=None,
            )
        with col2:
            source_rating = st.radio(
                "**The sources**",
                ["Right ones", "Some off", "Wrong ones"],
                key=f"source_rating_{turn_index}",
                index=None,
            )

        comment = st.text_area(
            "**Anything you'd add?**",
            key=f"comment_{turn_index}",
            placeholder=(
                "What was wrong, what was missing, whether it misread the "
                "literature, anything a philosopher would notice and I wouldn't."
            ),
            height=90,
        )

        if st.button("Send feedback", key=f"send_{turn_index}", type="primary"):
            if answer_rating is None and source_rating is None and not comment.strip():
                st.caption("Pick a rating or write something first.")
            else:
                log_feedback(
                    st.session_state.user_id,
                    turn.get("interaction_id", ""),
                    answer_rating or "",
                    source_rating or "",
                    comment.strip(),
                    turn["question"],
                )
                st.session_state.history[turn_index]["feedback_submitted"] = True
                st.rerun()


def looks_like_refusal(answer: str) -> bool:
    """
    True when the answer says the corpus had nothing to offer.

    Two conditions, both necessary. The answer has to be short, and it has
    to contain a phrase asserting an absence of information.

    The length check exists because marker matching alone kept producing
    false positives on *partial* answers. A multi-part question can have
    some parts the corpus covers and some it doesn't, and the model
    handles that well: it refuses the unanswerable parts and answers the
    rest from real passages. But refusal phrasing then appears inside an
    answer with several paragraphs of sourced content, and matching on
    markers alone suppressed the sources for all of it.

    Length is a crude proxy, but it targets the right distinction. A
    genuine refusal is a sentence or two, since there is nothing to say.
    Anything substantially longer has material behind it, and hiding the
    sources for that is worse than showing a few weak ones.

    This whole function is a stopgap. The principled version scores
    retrieved passages for relevance and refuses below a threshold, which
    needs similarity scores surfaced through the pipeline.
    """
    if len(answer) > REFUSAL_MAX_CHARS:
        return False
    lowered = answer.lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def render_sources(sources: list, subqueries: list, origins: list = None):
    """
    Sources grouped by the sub-question that retrieved them.

    Grouping rather than presenting a flat list because query_multi_concat()
    doesn't produce a globally ranked set: each sub-question contributes its
    own top matches, and those slices are concatenated. A flat list implies a
    ranking that isn't there, and on a multi-part question it buries whichever
    sub-question ran last. Grouping also makes it visible which half of a
    composite question each passage was actually answering, which is the thing
    worth asking testers about.

    Each passage shows an opening excerpt and expands to the full chunk, so the
    page stays readable while the complete text the model saw remains available.
    """
    if not sources:
        return

    # Drop heading-only and near-empty chunks. origins is filtered in step,
    # since it's positionally parallel to sources and the sub-question
    # grouping below would otherwise misattribute passages.
    if origins and len(origins) == len(sources):
        kept = [(s, o) for s, o in zip(sources, origins)
                if len(s.get("text", "") or "") >= MIN_SOURCE_CHARS]
        sources = [s for s, _ in kept]
        origins = [o for _, o in kept]
    else:
        sources = [s for s in sources
                   if len(s.get("text", "") or "") >= MIN_SOURCE_CHARS]

    if not sources:
        return

    # Fall back to a single ungrouped block if origin info isn't available,
    # e.g. results from an older cached session before this was added.
    if not origins or len(origins) != len(sources):
        with st.expander(f"Sources ({len(sources)} passages)"):
            for s in sources:
                render_single_source(s)
        return

    grouped = {}
    for src_item, origin in zip(sources, origins):
        grouped.setdefault(origin, []).append(src_item)

    with st.expander(f"Sources ({len(sources)} passages)"):
        multi = len(subqueries) > 1
        for sq_index in sorted(grouped):
            passages = grouped[sq_index]
            if multi:
                label = subqueries[sq_index] if sq_index < len(subqueries) else "Additional"
                count = f"{len(passages)} passage{'s' if len(passages) != 1 else ''}"
                # Tinted band rather than bold text: with five or six
                # sub-questions on a long answer, plain bold headings blur
                # into the passage text and the grouping stops being
                # legible at a glance, which is the whole point of it.
                st.markdown(
                    f"<div style='background:#eef2f7; border-left:4px solid #5c7cfa; "
                    f"padding:0.5em 0.9em; border-radius:4px; margin:0.9em 0 0.6em 0'>"
                    f"<span style='font-size:0.75em; letter-spacing:0.08em; "
                    f"text-transform:uppercase; color:#5c7cfa; font-weight:700'>"
                    f"Searched for</span><br>"
                    f"<span style='font-weight:600; color:#2c3e50'>{label}</span>"
                    f"<span style='font-size:0.8em; color:#7b8794'> &nbsp;·&nbsp; {count}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            for s in passages:
                render_single_source(s)
            if multi:
                st.divider()


def render_single_source(s: dict):
    """One passage: entry title, section, excerpt, expandable full text, link."""
    title = s.get("title", "unknown")
    section = s.get("section", "")
    url = s.get("source", "")
    text = s.get("text", "") or ""

    # Entry title carries the weight, section lighter beneath it.
    #
    # No hardcoded text colours here. An earlier version set the title to
    # near-black, which vanished against a dark theme. Streamlit's own
    # markdown inherits the active theme's foreground colour, and `opacity`
    # dims the section line relative to whatever that colour is rather than
    # against an assumed light background.
    #
    # Numbering is dropped: it implied a ranking that doesn't exist, since
    # each sub-question contributes its own top matches independently.
    st.markdown(
        f"<div style='font-size:1.05em; font-weight:700; "
        f"color:inherit; opacity:1; margin:0.7em 0 0.1em 0'>{title}</div>",
        unsafe_allow_html=True,
    )
    if section:
        st.markdown(
            f"<div style='opacity:0.65; font-size:0.9em; margin:0 0 0.4em 0'>"
            f"{section}</div>",
            unsafe_allow_html=True,
        )

    if text:
        # opacity rather than a fixed grey, and a semi-transparent border,
        # so both adapt to light and dark themes.
        style = ("font-size:0.88em; opacity:0.8; padding-left:1em; "
                 "border-left:2px solid rgba(128,128,128,0.35)")
        if len(text) <= 500:
            # Short enough to show whole; an expander here would just
            # repeat what's already on screen.
            st.markdown(f"<div style='{style}'>{text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='{style}'>{text[:400]}...</div>",
                unsafe_allow_html=True,
            )
            with st.expander("Full passage"):
                st.markdown(
                    f"<div style='{style}'>{text}</div>",
                    unsafe_allow_html=True,
                )

    if url:
        # st.markdown rather than st.caption: caption doesn't render links.
        st.markdown(f"<span style='font-size:0.8em'>[Read the full entry]({url})</span>",
                    unsafe_allow_html=True)
    st.write("")


# ---------------------------------------------------------------------

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if st.session_state.authenticated:
    main_app()
else:
    login_screen()