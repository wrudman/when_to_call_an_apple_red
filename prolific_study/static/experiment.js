/**************************************************************************
 * INITIALIZATION
 **************************************************************************/
const TEST_MODE = false;
const GOOGLE_FEEDBACK_URL =
  "https://docs.google.com/forms/d/e/1FAIpQLSeo9oxXp7fjrct1EQYdvnWMRFo8AahNQFZGtMyjYWnOfkNPhg/viewform?usp=header";


var jsPsych = initJsPsych({
  show_progress_bar: true,
  auto_update_progress_bar: false,
});

// ---------------------------------------------------------------------
// GLOBAL STATE
// ---------------------------------------------------------------------

let experimentEnded = false;
const MAX_DURATION_MS = 1000 * 30; // 30s test mode (use 30 * 60 * 1000 in prod)
let distractorErrors = 0; // global error tracking
let warningShown = false;

// ---------------------------------------------------------------------
// IMAGE HOSTING (Supabase)
// ---------------------------------------------------------------------

const SUPABASE_IMAGE_BASE =
  "https://utwhgfveotpusdjopcnl.supabase.co" +
  "/storage/v1/object/public/prolific_images/";

function preloadImage(url) {
  const img = new Image();
  img.src = url;
}

/**************************************************************************
 * Capture Prolific info
 **************************************************************************/

let subject_id = jsPsych.data.getURLVariable("PROLIFIC_PID");

if (!subject_id) {
  subject_id = getOrCreateTestPID();
}


const study_id = jsPsych.data.getURLVariable("STUDY_ID");
const session_id = jsPsych.data.getURLVariable("SESSION_ID");

jsPsych.data.addProperties({
  subject_id,
  study_id,
  session_id,
});

// ---------------------------------------------------------------------
// Timeline container
// ---------------------------------------------------------------------

let timeline = [];
TOTAL_QUESTIONS = 106;

/**************************************************************************
 * FETCH PROFILE FROM SERVER
 **************************************************************************/

function advanceProgress() {
  const cur = jsPsych.getProgressBarCompleted();
  jsPsych.setProgressBar(cur + 1 / TOTAL_QUESTIONS);
}


function getOrCreateTestPID() {
  const key = "TEST_PROLIFIC_PID";
  let pid = localStorage.getItem(key);

  if (!pid) {
    // Generate a Prolific-like ID (string, high entropy)
    pid = "TEST_" + crypto.randomUUID();
    localStorage.setItem(key, pid);
  }

  return pid;
}


async function fetchProfile() {
  const params = new URLSearchParams({ PROLIFIC_PID: subject_id });
  const response = await fetch(`/get_profile?${params.toString()}`);

  if (response.status === 403) {
    const payload = await response.json();

    sessionStorage.setItem(
      "exit_message",
      "You have already participated in this study."
    );

    window.location.href = "exit_return.html";
    throw new Error("Participant blocked");
  }

  if (!response.ok) throw new Error("Failed to fetch profile");
  return await response.json();
}


/**************************************************************************
 * SAFE TERMINATION
 **************************************************************************/

async function saveResults(exit_reason) {
  try {
    jsPsych.data.addProperties({
      exit_reason,
      exit_time: Date.now(),
    });

    await fetch("/save_results", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        PROLIFIC_PID: subject_id,
        data: jsPsych.data.get().values(),
      }),
    });
  } catch (err) {
    console.error("Failed to save results:", err);
  }
}

async function safeEndExperiment(message, reason, finalTrialData = null) {
  if (experimentEnded) return;
  experimentEnded = true;

  if (finalTrialData) {
    jsPsych.data.write(finalTrialData);
  }

  await saveResults(reason);

  sessionStorage.setItem("exit_message", message);
  sessionStorage.setItem("exit_reason", reason);

  window.location.href = "exit_return.html";
}



/**************************************************************************
 * RENDERERS
 **************************************************************************/

function warningTrial(message) {
  return {
    type: jsPsychHtmlButtonResponse,
    stimulus: `
      <div style="
        max-width: 1000px;
        margin: 0 auto;
        text-align: center;
      ">
        <h3>Warning</h3>
        <p>${message}</p>
        <p>
          Please answer carefully. The study will end if this happens again.
        </p>
      </div>
    `,
    choices: ["Continue"],
    data: {
      task_type: "warning",
    },
  };
}



function warningNode() {
  return {
    timeline: [
      warningTrial("You selected an unreasonable color."),
    ],
    conditional_function: function () {
      return distractorErrors === 1 && !warningShown;
    },
    on_timeline_finish: function () {
      warningShown = true;
    },
  };
}


function renderColorJudgment(q) {
  const choices = shuffle(getColorAnswerOptions(q.target_color));
  const imgId = `stim-img-${jsPsych.randomization.randomID(8)}`;

  let selectedColor = null;
  let fistCertainty = null;
  let secondCertainty = null;

  let trialStartTime = null;
  let colorTime = null;
  let firstCertaintyTime = null;
  let secondCertaintyTime = null;
  let certainty_change_count = 0;
  let finishTime = null;


  return {
    type: jsPsychHtmlButtonResponse,

    stimulus: `
      <div style="max-width:700px; margin:0 auto; text-align:center;">

        <div style="
          display:flex;
          justify-content:center;
          align-items:center;
          width:100%;
          margin:20px 0;
        ">
          <img
            id="${imgId}"
            src="${SUPABASE_IMAGE_BASE}${q.image_path}"
            style="max-width:400px; display:none;"
          >
        </div>

        <div style="margin:16px 0; font-weight:bold;">
          What color is the ${q.object} in the image?
        </div>

        <div style="
          margin-top:10px;
          font-size:14px;
          color:#555;
          max-width:520px;
          margin-left:auto;
          margin-right:auto;
        ">
          Choose the color that best matches your own judgment.
          The answer is never both or neither.
        </div>

        <div id="color-buttons"
          style="
            display:flex;
            justify-content:center;
            gap:20px;
            margin:24px 0 28px 0;
          ">
          ${choices.map(c => `
            <button class="color-btn" data-color="${c}">
              ${c}
            </button>
          `).join("")}
        </div>

        <div id="certainty-container" style="display:none; margin-top:24px;">

          <div style="margin-bottom:8px; font-size:14px;">
            How certain are you in your assessment?
          </div>

          <!-- semantic anchors -->
          <div style="
            position:relative;
            max-width:420px;
            margin:0 auto 8px auto;
            font-size:12px;
            color:#666;
            height:16px;
          ">
            <span style="position:absolute; left:0;">Very uncertain</span>
            <span style="position:absolute; left:50%; transform:translateX(-50%);">
              Unsure
            </span>
            <span style="position:absolute; right:0;">Very certain</span>
          </div>

          <div id="certainty-dots"
            style="
              display:flex;
              justify-content:space-between;
              max-width:420px;
              margin:0 auto;
            ">
            ${Array.from({ length: 10 }, (_, i) => `
              <div style="text-align:center;">
                <div
                  class="certainty-dot"
                  data-value="${i + 1}"
                  style="
                    width:18px;
                    height:18px;
                    border-radius:50%;
                    border:2px solid #999;
                    cursor:pointer;
                    margin:0 auto 4px auto;
                  "
                ></div>
                <div style="font-size:11px; color:#555;">
                  ${i + 1}
                </div>
              </div>
            `).join("")}
          </div>
        </div>

        <div style="margin-top:28px;">
          <button
            id="next-btn"
            class="jspsych-btn"
            style="opacity:0.4;"
            disabled
          >
            Next
          </button>
        </div>

      </div>
    `,

    choices: [],

    data: {
      task_type: "color_judgment",
      object: q.object,
      stimulus_type: q.stimulus_type,
      percent_colored: q.percent_colored,
      variant_region: q.variant_region,
      target_color: q.target_color,
      image_path: q.image_path,
    },

    on_load: function () {
      trialStartTime = performance.now();
      const img = document.getElementById(imgId);
      const colorButtons = document.querySelectorAll(".color-btn");
      const nextBtn = document.getElementById("next-btn");

      img.onload = () => {
        img.style.display = "block";
      };

      // ---- color selection ----
      colorButtons.forEach(btn => {
        btn.onclick = () => {
          colorTime = performance.now() - trialStartTime;
          selectedColor = btn.dataset.color;

          colorButtons.forEach(b => {
            b.disabled = true;
            b.style.opacity = "0.4";
          });

          btn.style.opacity = "1";
          btn.style.border = "2px solid #1976d2";

          document.getElementById("certainty-container").style.display = "block";
        };
      });

      // ---- certainty selection ----
      document.querySelectorAll(".certainty-dot").forEach(dot => {
        dot.onclick = () => {
          if (firstCertaintyTime === null) {
              firstCertaintyTime = performance.now() - trialStartTime;
              firstCertainty = Number(dot.dataset.value);
          } else {
              secondCertaintyTime = performance.now() - trialStartTime;
              secondCertainty = Number(dot.dataset.value);
          }
          
          certainty_change_count += 1;

          document.querySelectorAll(".certainty-dot").forEach(d => {
            d.style.background = "transparent";
            d.style.borderColor = "#999";
          });

          dot.style.background = "#1976d2";
          dot.style.borderColor = "#1976d2";

          // preload next image early
          if (q._next_image_path) {
            preloadImage(SUPABASE_IMAGE_BASE + q._next_image_path);
          }

          nextBtn.disabled = false;
          nextBtn.style.opacity = "1";
        };
      });

      nextBtn.onclick = () => {
        finishTime = performance.now() - trialStartTime;
        const allowed_answers =
          q.variant_region === "BG"
            ? ["white"]
            : [q.target_color, "white"];

        const isDistractor = !allowed_answers.includes(selectedColor);

        if (isDistractor) {
          distractorErrors += 1;
          jsPsych.data.addProperties({ distractor_errors: distractorErrors });
        }

        const trialData = {
          task_type: "color_judgment",
          object: q.object,
          stimulus_type: q.stimulus_type,
          percent_colored: q.percent_colored,
          variant_region: q.variant_region,
          target_color: q.target_color,
          image_path: q.image_path,

          response_label: selectedColor,
          initial_certainty: firstCertainty,
          final_certainty: secondCertainty,
          certainty_change_count: certainty_change_count,
          is_distractor: isDistractor,
          distractor_errors: distractorErrors,
          color_time: colorTime,
          first_certainty_time: firstCertaintyTime,
          second_certainty_time: secondCertaintyTime,
          finish_time: finishTime,
        };

        if (isDistractor && distractorErrors >= 2) {
          safeEndExperiment(
            "You selected unreasonable colors two times.",
            "failed_distractor",
            trialData
          );
          return;
        }
        jsPsych.finishTrial(trialData);
      };
    },
    on_finish: function (data) {
      advanceProgress();
    },
  };
}


function renderSanity(q) {

  // -----------------------------
  // TEXT SANITY (no q.options)
  // -----------------------------
  if (!q.options) {
    return {
      type: jsPsychSurveyText,
      questions: [{ prompt: q.prompt, rows: 3 }],
      data: {
        task_type: "sanity",
        sanity_id: q.sanity_id,
        correct_response: q.correct_response,
      },

      on_load: function () {
        const form = document.querySelector("form");

        form.addEventListener("submit", (e) => {
          e.preventDefault(); // stop jsPsych auto-advance

          const textarea = document.querySelector("textarea");
          const response = (textarea.value || "").trim().toLowerCase();
          const correct = q.correct_response.toLowerCase();

          if (response !== correct) {
            safeEndExperiment(
              "You did not pass the attention check.",
              "failed_attention"
            );
            return;
          }

          jsPsych.finishTrial({
            response_label: response,
            passed: true,
          });

          advanceProgress();
        });
      },
    };
  }

  // -----------------------------
  // LIKERT SANITY (q.options exists)
  // -----------------------------
  return {
    type: jsPsychSurveyLikert,
    questions: [
      {
        prompt: q.prompt,
        labels: q.options,
        required: true,
      },
    ],
    data: {
      task_type: "sanity",
      sanity_id: q.sanity_id,
      correct_response: q.correct_response,
    },

    on_finish: function (data) {
      const selectedIndex = data.response.Q0;
      const selectedLabel = q.options[selectedIndex];

      data.response_index = selectedIndex;
      data.response_label = selectedLabel;
      data.passed = selectedLabel === q.correct_response;

      if (!data.passed) {
        safeEndExperiment(
          "You did not pass the attention check.",
          "failed_attention"
        );
        return;
      }

      advanceProgress();
    },
  };
}


function renderIntrospection(q) {
  return {
    type: jsPsychHtmlSliderResponse,
    stimulus: `
      <div style="width:700px; margin:0 auto; text-align:left;">
        ${q.prompt}

        <p style="text-align:center; font-size:24px; margin-top:20px;">
          Selected value: <b><span id="slider-value">50</span>%</b>
        </p>
      </div>
    `,
    min: q.min ?? 0,
    max: q.max ?? 100,
    start: 50,
    step: 1,
    labels: ["0%", "100%"],
    require_movement: true,

    on_load: function () {
      const slider = document.querySelector('input[type="range"]');
      const valueSpan = document.getElementById("slider-value");

      valueSpan.textContent = slider.value;

      slider.addEventListener("input", () => {
        valueSpan.textContent = slider.value;
      });
    },

    data: {
      task_type: "introspection",
    },

    on_finish: function () {
      advanceProgress();
    },
  };
}



/**************************************************************************
 * GLOBAL TIMEOUT
 **************************************************************************/

function startGlobalTimeout() {
  const start = Date.now();

  jsPsych.data.addProperties({
    experiment_start_time: start,
  });

  window.setTimeout(() => {
    jsPsych.data.addProperties({
      timed_out: true,
      timeout_time: Date.now(),
    });

    console.warn("Experiment time limit reached (not terminating).");
  }, MAX_DURATION_MS);
}

/**************************************************************************
 * BUILD TIMELINE
 **************************************************************************/

function buildTimeline(questions) {
  timeline = [];

  // Instructions + timer start
  timeline.push({
    type: jsPsychInstructions,
    pages: [
      `
      <div class="jspsych-content" style="width:900px;text-align:left;">
        <h2>Welcome!</h2>

        <p>
          You will see a series of images showing an object and answer questions about them.
        </p>

        <p>
          Please answer <b>carefully and accurately</b>. Base your answer only on what you see in the image. <br>
          Some images may look unusual. Please answer as intuitively as possible.
        </p>

        <p><b>Important:</b></p>
        <ul>
          <li>You will be removed if you select a total of two unreasonable colors.</li>
          <li>You will be removed if you fail an attention check question.</li>
          <li>The study should take you approximately <b>15 minutes</b>.</li>
        </ul>

        <p>Click <b>Next</b> to begin.</p>
      </div>
      `,
    ],
    show_clickable_nav: true,
    allow_backward: false,
    on_finish: startGlobalTimeout,
  });

  // Annotate each question with the next image path (for preloading)
  questions.forEach((q, i) => {
    if (questions[i + 1]?.image_path) {
      q._next_image_path = questions[i + 1].image_path;
    }
  });

  // Questions
  questions.forEach((q) => {
    if (q.question_type === "sanity") {
      timeline.push(renderSanity(q));
    } else if (q.question_type === "introspection") {
      timeline.push(renderIntrospection(q));
    } else {
      timeline.push(renderColorJudgment(q));
      timeline.push(warningNode());
    }
  });

  timeline.push({
  type: jsPsychCallFunction,
  func: async () => {
    await saveResults("completed");

    const finishUrl = new URL("finish.html", window.location.origin);

    if (TEST_MODE) {
      finishUrl.searchParams.set("test", "1");
      finishUrl.searchParams.set(
        "feedback",
        encodeURIComponent(GOOGLE_FEEDBACK_URL)
      );
    }

    // allow POST to complete before navigation
    setTimeout(() => {
      window.location.href = finishUrl.toString();
    }, 300);
  },
});

}

/**************************************************************************
 * START EXPERIMENT
 **************************************************************************/

async function run_experiment() {
  try {
    const payload = await fetchProfile();

    jsPsych.data.addProperties({
      profile_id: payload.profile_id,
      profile_index: payload.profile_index,
    });

    buildTimeline(payload.questions);
    if (TEST_MODE) {
      sessionStorage.setItem("test_mode", "true");
    }
    jsPsych.run(timeline);

  } catch (err) {
    alert("Error loading experiment. Please contact the researcher.");
    console.error(err);
  }
}