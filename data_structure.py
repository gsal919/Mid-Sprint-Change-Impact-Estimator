import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

class FiservDataGenerator:
    """
    Generates pseudo‑data that matches Fiserv's structure with realistic delays.
    """
    
    def __init__(self):
        # ------------------------------------------------------------
        # Release stages (8 stages, overlapping)
        # ------------------------------------------------------------
        self.release_stages = [
            {"name": "Requirement/Discovery", "duration_weeks": 4, "order": 1, "parties": "Client"},
            {"name": "Tech Solution & Kick Off", "duration_weeks": 4, "order": 2, "parties": "Fiserv & Client"},
            {"name": "Design", "duration_weeks": 6, "order": 3, "parties": "Fiserv & Client"},
            {"name": "Develop", "duration_weeks": 8, "order": 4, "parties": "Fiserv & Client"},
            {"name": "SIT", "duration_weeks": 8, "order": 5, "parties": "Fiserv & Client"},
            {"name": "CAT/UAT", "duration_weeks": 6, "order": 6, "parties": "Fiserv & Client"},
            {"name": "System Implement", "duration_weeks": 1, "order": 7, "parties": "Fiserv & Client"},
            {"name": "App Launch", "duration_weeks": 0.2, "order": 8, "parties": "Client"}
        ]
        
        # ------------------------------------------------------------
        # Teams (6 teams with capacities)
        # ------------------------------------------------------------
        self.teams = [
            {"team_id": 1, "name": "Scrum Team 1", "location": "NZ", "headcount": 6, "hours_per_week": 240, "type": "Scrum"},
            {"team_id": 2, "name": "Scrum Team 2", "location": "NZ", "headcount": 7, "hours_per_week": 280, "type": "Scrum"},
            {"team_id": 3, "name": "Scrum Team 3", "location": "NZ", "headcount": 8, "hours_per_week": 320, "type": "Scrum"},
            {"team_id": 4, "name": "Scrum Team 4", "location": "NZ", "headcount": 9, "hours_per_week": 360, "type": "Scrum"},
            {"team_id": 5, "name": "Scrum Team 5", "location": "NZ", "headcount": 10, "hours_per_week": 400, "type": "Scrum"},
            {"team_id": 6, "name": "Scrum Team 6", "location": "Overseas", "headcount": 3, "hours_per_week": 120, "type": "Scrum"}
        ]
        
        # ------------------------------------------------------------
        # Resource types (for completeness)
        # ------------------------------------------------------------
        self.resources = {
            "iOS Developer": {"count": 2, "skill": "iOS", "hourly_rate": 120},
            "Android Developer": {"count": 3, "skill": "Android", "hourly_rate": 120},
            "Platform Developer": {"count": 4, "skill": "Platform", "hourly_rate": 130},
            "QA Manual": {"count": 3, "skill": "Manual", "hourly_rate": 80},
            "QA Automation": {"count": 4, "skill": "Automation", "hourly_rate": 100},
            "QA Performance": {"count": 2, "skill": "Performance", "hourly_rate": 110},
            "Delivery Manager": {"count": 3, "skill": "Management", "hourly_rate": 150},
            "BA": {"count": 3, "skill": "Analysis", "hourly_rate": 110},
            "Scrum Master": {"count": 3, "skill": "Agile", "hourly_rate": 120},
            "Architect": {"count": 3, "skill": "Architecture", "hourly_rate": 160}
        }
        
        # ------------------------------------------------------------
        # Work item hierarchy templates (Epic, Feature, Business Story, User Story, Task)
        # ------------------------------------------------------------
        self.epic_templates = [
            {"name": "Accounts", "story_points": 53},
            {"name": "Payments", "story_points": 30},
            {"name": "Security", "story_points": 19},
            {"name": "Reporting", "story_points": 18},
            {"name": "Notifications", "story_points": 10}
        ]
        
        self.feature_templates = {
            "Accounts": [
                {"name": "User Registration", "story_points": 6},
                {"name": "User Profile", "story_points": 8},
                {"name": "Account Settings", "story_points": 8},
                {"name": "Cards", "story_points": 43}
            ],
            "Payments": [
                {"name": "Payment Gateway", "story_points": 9},
                {"name": "Transaction History", "story_points": 7},
                {"name": "Refunds", "story_points": 8},
                {"name": "Recurring Billing", "story_points": 6}
            ],
            "Security": [
                {"name": "Authentication", "story_points": 7},
                {"name": "Authorization", "story_points": 5},
                {"name": "Audit Logs", "story_points": 4},
                {"name": "Encryption", "story_points": 3}
            ],
            "Reporting": [
                {"name": "Dashboard", "story_points": 6},
                {"name": "Custom Reports", "story_points": 5},
                {"name": "Data Export", "story_points": 3},
                {"name": "Scheduled Reports", "story_points": 4}
            ],
            "Notifications": [
                {"name": "Email Service", "story_points": 4},
                {"name": "Push Notifications", "story_points": 3},
                {"name": "In-App Messages", "story_points": 2},
                {"name": "SMS Alerts", "story_points": 1}
            ]
        }
        
        self.business_story_templates = {
            "Cards": [
                {"name": "Add Card", "story_points": 25},
                {"name": "Remove Card", "story_points": 15},
                {"name": "Update Card", "story_points": 10},
                {"name": "View Cards", "story_points": 3}
            ],
            "User Registration": [
                {"name": "Sign Up", "story_points": 3},
                {"name": "Email Verification", "story_points": 2},
                {"name": "Password Reset", "story_points": 1}
            ],
            "Payment Gateway": [
                {"name": "Process Payment", "story_points": 6},
                {"name": "Payment Validation", "story_points": 2},
                {"name": "Payment Callback", "story_points": 1}
            ]
        }
        
        self.user_story_templates = {
            "Add Card": [
                {"name": "Display Card Menu", "story_points": 7},
                {"name": "Display Card Screen", "story_points": 5},
                {"name": "Make a transfer", "story_points": 4},
                {"name": "Review transfer", "story_points": 2},
                {"name": "Transaction history", "story_points": 3},
                {"name": "Send email", "story_points": 1},
                {"name": "Verify Email", "story_points": 3}
            ],
            "Process Payment": [
                {"name": "Payment Form", "story_points": 2},
                {"name": "Payment Confirmation", "story_points": 3},
                {"name": "Error Handling", "story_points": 1}
            ]
        }
        
        self.task_templates = {
            "Display Card Menu": [
                {"name": "Back end development task -1", "platform": "Backend"},
                {"name": "Back end development task -2", "platform": "Backend"},
                {"name": "Environment setup", "platform": "DevOps"},
                {"name": "iOS development task 1", "platform": "iOS"},
                {"name": "Android development task -1", "platform": "Android"},
                {"name": "QA iOS", "platform": "QA"},
                {"name": "QA Android", "platform": "QA"}
            ],
            "Display Card Screen": [
                {"name": "Back end development task -1", "platform": "Backend"},
                {"name": "iOS development task 1", "platform": "iOS"},
                {"name": "Android development task -1", "platform": "Android"},
                {"name": "QA iOS", "platform": "QA"},
                {"name": "QA Android", "platform": "QA"}
            ],
            "Make a transfer": [
                {"name": "Backend API development", "platform": "Backend"},
                {"name": "iOS UI implementation", "platform": "iOS"},
                {"name": "Android UI implementation", "platform": "Android"},
                {"name": "QA Testing", "platform": "QA"}
            ]
        }
        
        # ------------------------------------------------------------
        # Priorities and statuses
        # ------------------------------------------------------------
        self.priorities = ["Low", "Medium", "High", "Critical"]
        self.statuses = ["To Do", "In Progress", "In Review", "Blocked", "Done", "Closed"]
        
        # Data storage
        self.releases = []
        self.sprints = []
        self.work_items = []
        self.change_requests = []
    
    # -----------------------------------------------------------------
    # Helper: realistic effort hours for tasks (used in work item generation)
    # -----------------------------------------------------------------
    def _get_effort_hours_for_task(self, story_points):
        """Return (original_estimate, rework, actual) for a task (1-4 story points)."""
        original = random.randint(story_points*3, story_points*6)
        
        rework = random.randint(0, int(original * 0.2))
        actual = original + rework
        #actual = max(original + 1, actual)
        return original, rework, actual
    
 
    
    # -----------------------------------------------------------------
    # Generate releases (3 parallel)
    # -----------------------------------------------------------------
    def generate_releases(self, start_date=datetime(2024, 1, 1)):
        print("Generating releases...")
        for i in range(9):
            name = ["X+2", "X+3", "X+4", "X+5", "X+6", "X+7","X+8", "X+9", "X+10"][i]
            release = {
                "release_id": f"REL_{name}",
                "release_name": f"Release {name}",
                "start_date": start_date + timedelta(weeks=i*10),
                "stages": []
            }
            current = release["start_date"]
            for stage in self.release_stages:
                end = current + timedelta(weeks=stage["duration_weeks"])
                release["stages"].append({
                    "stage_name": stage["name"],
                    "start_date": current,
                    "end_date": end,
                    "duration_weeks": stage["duration_weeks"]
                })
                current = end
            release["end_date"] = current
            self.releases.append(release)
        print(f"  -> {len(self.releases)} releases")
    
    # -----------------------------------------------------------------
    # Generate sprints (2‑week sprints for each team)
    # -----------------------------------------------------------------
    def generate_sprints(self):
        print("Generating sprints...")
        sprint_id = 1
        for release in self.releases:
            for team in self.teams:
                duration_days = (release["end_date"] - release["start_date"]).days
                num_sprints = max(1, duration_days // 10)
                for sn in range(1, num_sprints+1):
                    start = release["start_date"] + timedelta(days=(sn-1)*10)
                    end = start + timedelta(days=10)
                    if end <= release["end_date"]:
                        sprint = {
                            "sprint_id": sprint_id,
                            "sprint_name": f"Sprint {sn} - {team['name']} - {release['release_name']}",
                            "sprint_number": sn,
                            "team_id": team["team_id"],
                            "team_name": team["name"],
                            "release_id": release["release_id"],
                            "start_date": start,
                            "end_date": end,
                            "capacity_hours": team["hours_per_week"],
                            "headcount": team["headcount"]
                        }
                        self.sprints.append(sprint)
                        sprint_id += 1
        print(f"  -> {len(self.sprints)} sprints")
    
    # -----------------------------------------------------------------
    # Generate work item hierarchy (Epic → Feature → BS → US → Tasks)
    # -----------------------------------------------------------------
    def generate_work_items(self, target_tasks=5000):
        print("Generating work items...")
        item_id = 1
        task_count = 0
        
        # Flatten templates for random sampling
        all_features = sum(self.feature_templates.values(), [])
        all_bs = sum(self.business_story_templates.values(), [])
        all_us = sum(self.user_story_templates.values(), [])
        all_tasks_flat = [t for sublist in self.task_templates.values() for t in sublist]
        
        # Story point distribution for User Stories (realistic)
        us_sp = random.randint(3, 13)
        
        while task_count < target_tasks:
            # Epic
            epic = random.choice(self.epic_templates)
            release = random.choice(self.releases)
            epic_item = {
                "work_item_id": item_id, "level": "Epic", "name": epic["name"],
                "story_points": epic["story_points"], "parent_id": None,
                "release_id": release["release_id"], "priority": random.choice(self.priorities),
                "status": random.choice(self.statuses)
            }
            self.work_items.append(epic_item)
            item_id += 1
            
            # Features
            num_feat = random.randint(2,4)
            features = random.sample(all_features, min(num_feat, len(all_features)))
            for ft in features:
                feature_item = {
                    "work_item_id": item_id, "level": "Feature", "name": ft["name"],
                    "story_points": ft["story_points"], "parent_id": epic_item["work_item_id"],
                    "release_id": release["release_id"], "priority": epic_item["priority"],
                    "status": random.choice(self.statuses)
                }
                self.work_items.append(feature_item)
                item_id += 1
                
                # Business Stories
                num_bs = random.randint(2,3)
                bs_list = random.sample(all_bs, min(num_bs, len(all_bs)))
                for bs in bs_list:
                    bs_item = {
                        "work_item_id": item_id, "level": "Business Story", "name": bs["name"],
                        "story_points": bs["story_points"], "parent_id": feature_item["work_item_id"],
                        "release_id": release["release_id"], "priority": feature_item["priority"],
                        "status": random.choice(self.statuses)
                    }
                    self.work_items.append(bs_item)
                    item_id += 1
                    
                    # User Stories
                    num_us = random.randint(2,4)
                    us_list = random.sample(all_us, min(num_us, len(all_us)))
                    for us_template in us_list:
                        #us_sp = np.random.choice(us_sp_choices, p=us_sp_weights)
                        us_sp = random.randint(3, 13)
                        us_item = {
                            "work_item_id": item_id, "level": "User Story", "name": us_template["name"],
                            "story_points": us_sp, "parent_id": bs_item["work_item_id"],
                            "release_id": release["release_id"], "priority": bs_item["priority"],
                            "status": random.choice(self.statuses)
                        }
                        self.work_items.append(us_item)
                        item_id += 1
                        
                        # Tasks
                        sprint = random.choice(self.sprints)
                        num_tasks = random.randint(3,7)
                        for _ in range(num_tasks):
                            t = random.choice(all_tasks_flat)
                            task_sp = random.choice([1,2,3,4])
                            orig_est, rework, actual = self._get_effort_hours_for_task(task_sp)
                            task_item = {
                                "work_item_id": item_id, "level": "Task", "name": t["name"],
                                "story_points": task_sp, "parent_id": us_item["work_item_id"],
                                "release_id": release["release_id"], "sprint_id": sprint["sprint_id"],
                                "team_id": sprint["team_id"], "platform": t["platform"],
                                "priority": us_item["priority"], "status": random.choice(self.statuses),
                                "original_estimate_hours": orig_est, "rework_hours": rework,
                                "actual_hours": actual
                            }
                            self.work_items.append(task_item)
                            item_id += 1
                            task_count += 1
                            if task_count >= target_tasks:
                                break
                        if task_count >= target_tasks:
                            break
                    if task_count >= target_tasks:
                        break
                if task_count >= target_tasks:
                    break
            if task_count >= target_tasks:
                break
        
        print(f"  -> {len(self.work_items)} work items (tasks: {task_count})")
    
    # -----------------------------------------------------------------
    # Generate change requests (mid‑sprint) – realistic delays
    # -----------------------------------------------------------------
    def generate_change_requests(self, num_changes=50000):
        print(f"Generating {num_changes} change requests...")
        changes = []

        # Pre‑compute average sprint task load
        tasks_per_sprint = {}
        for w in self.work_items:
            if w["level"] == "Task" and "sprint_id" in w:
                sid = w["sprint_id"]
                tasks_per_sprint[sid] = tasks_per_sprint.get(sid, 0) + 1
        avg_task_load = np.mean(list(tasks_per_sprint.values())) if tasks_per_sprint else 5

        for i in range(num_changes):
            sprint = random.choice(self.sprints)
            candidates = [w for w in self.work_items if w["level"] in ["User Story", "Task"]]
            if not candidates:
                candidates = self.work_items
            work_item = random.choice(candidates)

            story_pts = work_item["story_points"]

            # ============================================================
            # 1. ORIGINAL ESTIMATE (hours) based on story points
            # ============================================================
            original = random.randint(story_pts * 4, story_pts * 7)

            # ============================================================
            # 2. ACTUAL HOURS (including rework) based on Fiserv patterns
            # ============================================================
            rework = random.randint(0, int(original * 0.3))
            actual = random.randint(original, original + rework)

            actual = max(1, actual)
            
            base_effort = actual

        

            # ============================================================
            # 3. Sprint timing & effort multipliers
            # ============================================================
            days_into = random.randint(1, 9)
            sprint_progress = days_into / 10
            timing_penalty = 1 + (sprint_progress * 0.7)

            priority_factor = {"Low":0.3, "Medium":0.6, "High":0.9, "Critical":1.2}.get(work_item["priority"], 1.0)
            complexity_multiplier = 1 + (story_pts / 15)
            #dependency_multiplier = random.uniform(1.0, 2.5)

            impacted = (
                base_effort 
                * priority_factor 
                * timing_penalty 
                * complexity_multiplier 
                #* dependency_multiplier
            )

            # ============================================================
            # 4. Remaining capacity with random utilisation
            # ============================================================
            remaining_base = sprint["capacity_hours"] * (1 - sprint_progress)
            utilisation = random.uniform(0.6, 0.8)   # team already 75-97% busy
            remaining = remaining_base * (1 - utilisation)

            # ============================================================
            # 5. Spillover decision
            # ============================================================
            if story_pts <= 3:
                base_spill_prob = 0.30
            elif story_pts <= 5:
                base_spill_prob = 0.40
            elif story_pts <= 8:
                base_spill_prob = 0.70
            else:
                base_spill_prob = 0.90

            capacity_ratio = impacted / max(remaining, 1)
            spillover_prob = min(0.99, base_spill_prob * capacity_ratio)
            spillover = random.random() < spillover_prob

            # ============================================================
            # 6. Delay days calculation
            # ============================================================
            if spillover:
                daily_output = sprint["headcount"] * random.uniform(0.6, 1.0)
                extra_effort = max(impacted - remaining, 0)
                overload_ratio = min(impacted / max(remaining, 20), 5)
                base_delay = extra_effort / max(daily_output, 1)
                coordination_penalty = 1 + np.log1p(overload_ratio)
                delay_days = base_delay * coordination_penalty

                # Heavy tail but controlled (8% chance of extreme delay)
                if random.random() < 0.08:
                    delay_days *= random.uniform(1.5, 3)
                delay_days = round(min(delay_days, 45), 1)
            else:
                delay_days = 0.0

            # ============================================================
            # 7. Severity based on delay days
            # ============================================================
            if delay_days == 0:
                severity = "On Time"
            elif delay_days <= 5:
                severity = "Minor"
            elif delay_days <= 20:
                severity = "Moderate"
            elif delay_days <= 44:
                severity = "Severe"
            else:
                severity = "Critical"

            # ============================================================
            # 8. Additional ML features
            # ============================================================
            complexity_score = story_pts * (work_item.get("affected_components", 1))
            predicted_risk_proxy = story_pts * 0.5 + (work_item.get("affected_components", 1)) * 0.3 + sprint_progress * 10
            sprint_task_load = tasks_per_sprint.get(sprint["sprint_id"], avg_task_load)

            # Build change request record
            change = {
                "change_request_id": f"CR_{i+1:06d}",
                "sprint_id": sprint["sprint_id"],
                "sprint_name": sprint["sprint_name"],
                "release_id": sprint["release_id"],
                "work_item_id": work_item["work_item_id"],
                "work_item_name": work_item["name"],
                "work_item_level": work_item["level"],
                "item_type": work_item["level"],
                "story_points": story_pts,
                "priority": work_item["priority"],
                "request_date": sprint["start_date"] + timedelta(days=days_into),
                "days_into_sprint": days_into,
                "sprint_progress_pct": round(sprint_progress * 100, 1),
                "original_estimate_hours": original,
                "rework_hours": rework,
                "actual_hours": actual,
                "base_effort_hours": round(base_effort, 1),
                "impacted_effort_hours": round(impacted, 1),
                "remaining_capacity_hours": round(remaining, 1),
                "caused_spillover": 1 if spillover else 0,
                "delay_days_caused": delay_days,
                "delay_severity": severity,
                "complexity_score": round(complexity_score, 2),
                "predicted_risk_proxy": round(predicted_risk_proxy, 2),
                "sprint_task_load": sprint_task_load,
                "timing_penalty": round(timing_penalty, 2),
                "priority_multiplier": priority_factor,
                "recommendation": self._get_recommendation(spillover, sprint_progress, priority_factor)
            }
            changes.append(change)

        df = pd.DataFrame(changes)
        print(f"  -> Generated {len(df)} change requests")
        print(f"     Spillover rate: {df['caused_spillover'].mean():.1%}")
        print(f"     Avg delay days: {df['delay_days_caused'].mean():.2f}")
        print(f"     Delay severity distribution:\n{df['delay_severity'].value_counts()}")
        return df
    
    def _get_recommendation(self, spillover, progress, priority_factor):
        if not spillover:
            return "Accept in current sprint"
        if priority_factor > 1.3:
            return "Accept with schedule adjustment"
        if progress > 0.7:
            return "Defer to next sprint"
        return "Split scope or escalate"
    
    # -----------------------------------------------------------------
    # Generate all data and save to CSV
    # -----------------------------------------------------------------
    def generate_all(self, output_dir="fiserv_data", num_changes=50000, target_tasks=5000):
        os.makedirs(output_dir, exist_ok=True)
        self.generate_releases()
        self.generate_sprints()
        self.generate_work_items(target_tasks=target_tasks)
        change_df = self.generate_change_requests(num_changes=num_changes)
        
        # Save
        releases_df = pd.DataFrame(self.releases).drop(columns=["stages"], errors="ignore")
        sprints_df = pd.DataFrame(self.sprints)
        work_items_df = pd.DataFrame(self.work_items)
        
        releases_df.to_csv(f"{output_dir}/releases.csv", index=False)
        sprints_df.to_csv(f"{output_dir}/sprints.csv", index=False)
        work_items_df.to_csv(f"{output_dir}/work_items.csv", index=False)
        change_df.to_csv(f"{output_dir}/change_requests.csv", index=False)
        
        # Teams and resources (static)
        pd.DataFrame(self.teams).to_csv(f"{output_dir}/teams.csv", index=False)
        res_df = pd.DataFrame([{"role":k, "count":v["count"], "skill":v["skill"]} for k,v in self.resources.items()])
        res_df.to_csv(f"{output_dir}/resource_types.csv", index=False)
        
        print(f"\n✅ All data saved to '{output_dir}/'")
        return change_df

if __name__ == "__main__":
    gen = FiservDataGenerator()
    changes = gen.generate_all(num_changes=50000, target_tasks=5000)
