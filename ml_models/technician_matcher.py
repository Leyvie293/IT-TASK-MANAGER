"""
Technician Matcher Model
Matches tasks to the most suitable technician based on skills, workload, and history
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime

class TechnicianMatcher:
    """Matches tasks to optimal technicians"""
    
    def __init__(self, model_path='ml_models/technician_matcher.pkl'):
        self.model_path = model_path
        self.scaler = StandardScaler()
        self.technician_encoder = LabelEncoder()
        self.skill_encoder = LabelEncoder()
        self.knn_model = None
        self.technician_profiles = {}
        self.is_trained = False
        
    def create_technician_profile(self, technician):
        """
        Create a profile vector for a technician
        
        Args:
            technician: Dictionary with technician data
            
        Returns:
            Profile vector
        """
        profile = {}
        
        # Skills (one-hot encoded)
        skills = technician.get('skills', [])
        if isinstance(skills, str):
            skills = [s.strip() for s in skills.split(',')]
        
        profile['skills_count'] = len(skills)
        
        # Experience
        profile['experience_months'] = technician.get('experience_months', 0)
        
        # Workload
        profile['current_tasks'] = technician.get('current_tasks', 0)
        profile['max_tasks'] = technician.get('max_tasks', 5)
        profile['workload_ratio'] = profile['current_tasks'] / max(profile['max_tasks'], 1)
        
        # Performance metrics
        profile['avg_resolution_time'] = technician.get('avg_resolution_time', 24)
        profile['sla_compliance_rate'] = technician.get('sla_compliance_rate', 0.8)
        profile['task_completion_rate'] = technician.get('task_completion_rate', 0.9)
        
        # Availability
        profile['is_available'] = 1 if technician.get('is_available', True) else 0
        
        # Specialty scores (based on historical performance by category)
        specialties = technician.get('specialties', {})
        for category in ['Network', 'Hardware', 'Software', 'Security', 'Email']:
            profile[f'specialty_{category}'] = specialties.get(category, 0.5)
        
        return profile
    
    def create_task_profile(self, task, category_importance=None):
        """
        Create a profile vector for a task
        
        Args:
            task: Dictionary with task data
            category_importance: Dictionary mapping categories to importance scores
            
        Returns:
            Task profile vector
        """
        profile = {}
        
        # Category requirements
        category = task.get('category', 'General')
        if category_importance:
            for cat, importance in category_importance.items():
                profile[f'requires_{cat}'] = 1 if category == cat else 0
                profile[f'importance_{cat}'] = importance if category == cat else 0
        else:
            profile[f'requires_{category}'] = 1
        
        # Priority
        priority = task.get('priority', 'Medium')
        priority_map = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1}
        profile['priority_score'] = priority_map.get(priority, 2)
        
        # Complexity
        complexity = task.get('complexity', 'Medium')
        complexity_map = {'Complex': 3, 'Medium': 2, 'Simple': 1}
        profile['complexity_score'] = complexity_map.get(complexity, 2)
        
        # Time sensitivity
        profile['has_sla'] = 1 if task.get('sla_due_date') else 0
        
        # Required skills (extracted from description)
        required_skills = self._extract_required_skills(task)
        profile['required_skills_count'] = len(required_skills)
        
        # Department
        department = task.get('department', 'General')
        profile[f'department_{department}'] = 1
        
        return profile, required_skills
    
    def _extract_required_skills(self, task):
        """Extract required skills from task description"""
        skills_keywords = {
            'Network': ['wifi', 'router', 'switch', 'firewall', 'vpn', 'dns', 'ip'],
            'Hardware': ['printer', 'computer', 'laptop', 'monitor', 'keyboard', 'mouse'],
            'Software': ['windows', 'office', 'outlook', 'install', 'license', 'update'],
            'Security': ['camera', 'access', 'biometric', 'alarm', 'surveillance'],
            'Database': ['sql', 'database', 'backup', 'restore', 'query'],
            'Programming': ['python', 'java', 'script', 'code', 'development'],
            'Email': ['outlook', 'email', 'mail', 'spam', 'inbox']
        }
        
        text = (task.get('title', '') + ' ' + task.get('description', '')).lower()
        required_skills = []
        
        for skill, keywords in skills_keywords.items():
            if any(keyword in text for keyword in keywords):
                required_skills.append(skill)
        
        return list(set(required_skills))
    
    def calculate_skill_match_score(self, technician_skills, required_skills):
        """Calculate skill match score between technician and task"""
        if not required_skills:
            return 1.0
        
        if isinstance(technician_skills, str):
            technician_skills = [s.strip() for s in technician_skills.split(',')]
        
        # Count matching skills
        matching_skills = set(technician_skills) & set(required_skills)
        
        if not required_skills:
            return 0.0
        
        return len(matching_skills) / len(required_skills)
    
    def train(self, historical_assignments, technician_data):
        """
        Train the matching model
        
        Args:
            historical_assignments: List of successful task assignments
            technician_data: List of technician profiles
        """
        print("Training Technician Matcher...")
        
        # Create technician profiles
        self.technician_profiles = {}
        technician_vectors = []
        technician_ids = []
        
        for tech in technician_data:
            tech_id = tech['id']
            profile = self.create_technician_profile(tech)
            self.technician_profiles[tech_id] = {
                'profile': profile,
                'data': tech
            }
            technician_vectors.append(list(profile.values()))
            technician_ids.append(tech_id)
        
        # Convert to numpy array
        X = np.array(technician_vectors)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train KNN model
        self.knn_model = NearestNeighbors(
            n_neighbors=min(5, len(technician_ids)),
            metric='cosine',
            algorithm='auto'
        )
        self.knn_model.fit(X_scaled)
        
        # Encode technician IDs
        self.technician_encoder.fit(technician_ids)
        
        self.is_trained = True
        self.save_model()
        
        print(f"Trained on {len(technician_ids)} technicians")
        
        return True
    
    def find_best_match(self, task, available_technicians=None, top_n=3):
        """
        Find best matching technicians for a task
        
        Args:
            task: Task dictionary
            available_technicians: List of available technician IDs
            top_n: Number of top matches to return
            
        Returns:
            List of matched technicians with scores
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Train or load model first.")
        
        if not self.technician_profiles:
            raise ValueError("No technician profiles loaded.")
        
        # Filter available technicians
        if available_technicians is None:
            available_technicians = list(self.technician_profiles.keys())
        
        # Create task profile
        task_profile, required_skills = self.create_task_profile(task)
        
        # Prepare comparison data
        technician_scores = []
        
        for tech_id in available_technicians:
            if tech_id not in self.technician_profiles:
                continue
            
            tech_data = self.technician_profiles[tech_id]
            tech_profile = tech_data['profile']
            
            # Calculate various match scores
            scores = {}
            
            # 1. Skill match score
            tech_skills = tech_data['data'].get('skills', [])
            scores['skill_match'] = self.calculate_skill_match_score(tech_skills, required_skills)
            
            # 2. Workload score (prefer less busy technicians)
            workload_ratio = tech_profile.get('workload_ratio', 0)
            scores['workload_score'] = 1 - workload_ratio  # Lower workload = higher score
            
            # 3. Experience score
            experience_months = tech_profile.get('experience_months', 0)
            scores['experience_score'] = min(experience_months / 60, 1)  # Cap at 5 years
            
            # 4. Performance score
            sla_rate = tech_profile.get('sla_compliance_rate', 0.8)
            completion_rate = tech_profile.get('task_completion_rate', 0.9)
            scores['performance_score'] = (sla_rate + completion_rate) / 2
            
            # 5. Category specialty score
            category = task.get('category', 'General')
            specialty_key = f'specialty_{category}'
            scores['specialty_score'] = tech_profile.get(specialty_key, 0.5)
            
            # 6. Availability score
            scores['availability_score'] = tech_profile.get('is_available', 1)
            
            # Calculate weighted total score
            weights = {
                'skill_match': 0.3,
                'workload_score': 0.2,
                'experience_score': 0.15,
                'performance_score': 0.15,
                'specialty_score': 0.1,
                'availability_score': 0.1
            }
            
            total_score = sum(scores[key] * weights[key] for key in weights)
            
            technician_scores.append({
                'technician_id': tech_id,
                'technician_name': tech_data['data'].get('name', 'Unknown'),
                'total_score': total_score,
                'scores': scores,
                'skills': tech_skills,
                'current_workload': tech_profile.get('current_tasks', 0),
                'max_workload': tech_profile.get('max_tasks', 5),
                'experience_months': experience_months,
                'specialty': category,
                'specialty_score': scores['specialty_score']
            })
        
        # Sort by total score
        technician_scores.sort(key=lambda x: x['total_score'], reverse=True)
        
        # Return top N matches
        return technician_scores[:top_n]
    
    def recommend_assignments(self, tasks, technicians, strategy='balanced'):
        """
        Recommend assignments for multiple tasks
        
        Args:
            tasks: List of task dictionaries
            technicians: List of technician dictionaries
            strategy: 'balanced', 'skill_focused', or 'fast_turnaround'
            
        Returns:
            Assignment recommendations
        """
        # Update weights based on strategy
        weight_configs = {
            'balanced': {
                'skill_match': 0.3,
                'workload_score': 0.2,
                'experience_score': 0.15,
                'performance_score': 0.15,
                'specialty_score': 0.1,
                'availability_score': 0.1
            },
            'skill_focused': {
                'skill_match': 0.5,
                'workload_score': 0.1,
                'experience_score': 0.15,
                'performance_score': 0.15,
                'specialty_score': 0.05,
                'availability_score': 0.05
            },
            'fast_turnaround': {
                'skill_match': 0.2,
                'workload_score': 0.3,
                'experience_score': 0.1,
                'performance_score': 0.2,
                'specialty_score': 0.1,
                'availability_score': 0.1
            }
        }
        
        weights = weight_configs.get(strategy, weight_configs['balanced'])
        
        # Train if not already trained
        if not self.is_trained:
            self.train([], technicians)
        
        assignments = []
        assigned_tasks = set()
        technician_load = {tech['id']: 0 for tech in technicians}
        
        # Sort tasks by priority
        priority_order = {'Critical': 4, 'High': 3, 'Medium': 2, 'Low': 1}
        tasks_sorted = sorted(tasks, key=lambda x: priority_order.get(x.get('priority', 'Medium'), 2), reverse=True)
        
        for task in tasks_sorted:
            task_id = task.get('id', hash(str(task)))
            
            # Filter technicians who can take more tasks
            available_techs = []
            for tech in technicians:
                tech_id = tech['id']
                max_tasks = tech.get('max_tasks', 5)
                if technician_load[tech_id] < max_tasks:
                    available_techs.append(tech_id)
            
            if not available_techs:
                assignments.append({
                    'task_id': task_id,
                    'task_title': task.get('title', 'Unknown'),
                    'assigned_to': None,
                    'reason': 'No available technicians',
                    'scores': {}
                })
                continue
            
            # Find best match
            matches = self.find_best_match(task, available_techs, top_n=len(available_techs))
            
            if matches:
                # Select best match considering current load
                best_match = None
                best_score = -1
                
                for match in matches:
                    tech_id = match['technician_id']
                    
                    # Adjust score based on current load
                    load_penalty = technician_load[tech_id] / match['max_workload']
                    adjusted_score = match['total_score'] * (1 - load_penalty * 0.3)
                    
                    if adjusted_score > best_score:
                        best_score = adjusted_score
                        best_match = match
                
                if best_match:
                    tech_id = best_match['technician_id']
                    technician_load[tech_id] += 1
                    assigned_tasks.add(task_id)
                    
                    assignments.append({
                        'task_id': task_id,
                        'task_title': task.get('title', 'Unknown'),
                        'assigned_to': tech_id,
                        'technician_name': best_match['technician_name'],
                        'match_score': best_match['total_score'],
                        'adjusted_score': best_score,
                        'scores': best_match['scores'],
                        'reason': f"Best match based on {strategy} strategy"
                    })
        
        # Calculate assignment statistics
        assigned_count = len([a for a in assignments if a['assigned_to'] is not None])
        unassigned_count = len(assignments) - assigned_count
        
        return {
            'assignments': assignments,
            'statistics': {
                'total_tasks': len(tasks),
                'assigned_tasks': assigned_count,
                'unassigned_tasks': unassigned_count,
                'assignment_rate': assigned_count / len(tasks) if tasks else 0,
                'strategy_used': strategy,
                'workload_distribution': technician_load
            }
        }
    
    def save_model(self):
        """Save model to disk"""
        model_data = {
            'scaler': self.scaler,
            'technician_encoder': self.technician_encoder,
            'skill_encoder': self.skill_encoder,
            'knn_model': self.knn_model,
            'technician_profiles': self.technician_profiles,
            'is_trained': self.is_trained
        }
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {self.model_path}")
    
    def load_model(self):
        """Load model from disk"""
        import os
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.scaler = model_data['scaler']
            self.technician_encoder = model_data['technician_encoder']
            self.skill_encoder = model_data['skill_encoder']
            self.knn_model = model_data['knn_model']
            self.technician_profiles = model_data['technician_profiles']
            self.is_trained = model_data['is_trained']
            
            print(f"Model loaded from {self.model_path}")
            return True
        else:
            print(f"No model found at {self.model_path}")
            return False
    
    def update_technician_profiles(self, new_technician_data):
        """Update technician profiles with new data"""
        for tech in new_technician_data:
            tech_id = tech['id']
            if tech_id in self.technician_profiles:
                # Update existing profile
                self.technician_profiles[tech_id]['data'].update(tech)
                self.technician_profiles[tech_id]['profile'] = self.create_technician_profile(tech)
            else:
                # Add new technician
                profile = self.create_technician_profile(tech)
                self.technician_profiles[tech_id] = {
                    'profile': profile,
                    'data': tech
                }
        
        # Retrain if significant changes
        if len(new_technician_data) > 0:
            print(f"Updated {len(new_technician_data)} technician profiles")
    
    def generate_technician_data_from_database(self, db_session):
        """
        Generate technician data from database
        
        Args:
            db_session: SQLAlchemy session
            
        Returns:
            List of technician data
        """
        from models.user_models import User
        from models.task_models import Task
        from datetime import datetime, timedelta
        
        # Get all technicians
        technicians = db_session.query(User).filter(
            User.role.in_(['Technician', 'Supervisor'])
        ).all()
        
        technician_data = []
        
        for tech in technicians:
            # Calculate performance metrics
            completed_tasks = db_session.query(Task).filter(
                Task.assigned_to == tech.id,
                Task.status == 'Closed'
            ).all()
            
            total_tasks = len(completed_tasks)
            
            if total_tasks > 0:
                # Calculate average resolution time
                total_time = 0
                sla_met_count = 0
                
                for task in completed_tasks:
                    if task.start_time and task.end_time:
                        resolution_hours = (task.end_time - task.start_time).total_seconds() / 3600
                        total_time += resolution_hours
                    
                    if task.sla_due_date and task.end_time:
                        if task.end_time <= task.sla_due_date:
                            sla_met_count += 1
                
                avg_resolution_time = total_time / total_tasks if total_tasks > 0 else 24
                sla_compliance_rate = sla_met_count / total_tasks if total_tasks > 0 else 0.8
            else:
                avg_resolution_time = 24
                sla_compliance_rate = 0.8
            
            # Calculate category specialties
            specialties = {}
            categories = ['Network', 'Hardware', 'Software', 'Security', 'Email']
            
            for category in categories:
                cat_tasks = db_session.query(Task).filter(
                    Task.assigned_to == tech.id,
                    Task.category == category,
                    Task.status == 'Closed'
                ).all()
                
                if len(cat_tasks) > 0:
                    # Calculate success rate for this category
                    successful_tasks = len([t for t in cat_tasks if t.end_time and t.sla_due_date and t.end_time <= t.sla_due_date])
                    specialties[category] = successful_tasks / len(cat_tasks)
                else:
                    specialties[category] = 0.5  # Default
            
            # Current workload
            current_tasks = db_session.query(Task).filter(
                Task.assigned_to == tech.id,
                Task.status.in_(['Assigned', 'In Progress'])
            ).count()
            
            # Estimate experience (based on join date)
            experience_months = 12  # Default
            if tech.created_at:
                experience_months = max(1, (datetime.utcnow() - tech.created_at).days // 30)
            
            tech_data = {
                'id': tech.id,
                'name': tech.full_name,
                'skills': tech.skills or [],
                'experience_months': experience_months,
                'current_tasks': current_tasks,
                'max_tasks': tech.max_tasks or 5,
                'avg_resolution_time': avg_resolution_time,
                'sla_compliance_rate': sla_compliance_rate,
                'task_completion_rate': 0.9,  # Could calculate from history
                'is_available': tech.is_available,
                'specialties': specialties,
                'department': tech.department
            }
            
            technician_data.append(tech_data)
        
        return technician_data

# Test the matcher
if __name__ == "__main__":
    # Create sample technician data
    technicians = [
        {
            'id': 'tech1',
            'name': 'John Network Specialist',
            'skills': ['Network', 'Security', 'Hardware'],
            'experience_months': 36,
            'current_tasks': 2,
            'max_tasks': 5,
            'avg_resolution_time': 6.5,
            'sla_compliance_rate': 0.95,
            'task_completion_rate': 0.98,
            'is_available': True,
            'specialties': {
                'Network': 0.95,
                'Hardware': 0.85,
                'Software': 0.65,
                'Security': 0.90,
                'Email': 0.70
            }
        },
        {
            'id': 'tech2',
            'name': 'Sarah Software Expert',
            'skills': ['Software', 'Email', 'Database'],
            'experience_months': 24,
            'current_tasks': 4,
            'max_tasks': 5,
            'avg_resolution_time': 8.2,
            'sla_compliance_rate': 0.88,
            'task_completion_rate': 0.92,
            'is_available': True,
            'specialties': {
                'Network': 0.70,
                'Hardware': 0.75,
                'Software': 0.95,
                'Security': 0.60,
                'Email': 0.90
            }
        },
        {
            'id': 'tech3',
            'name': 'Mike Security Analyst',
            'skills': ['Security', 'Network', 'Programming'],
            'experience_months': 18,
            'current_tasks': 1,
            'max_tasks': 5,
            'avg_resolution_time': 12.5,
            'sla_compliance_rate': 0.82,
            'task_completion_rate': 0.88,
            'is_available': True,
            'specialties': {
                'Network': 0.80,
                'Hardware': 0.70,
                'Software': 0.75,
                'Security': 0.92,
                'Email': 0.65
            }
        },
        {
            'id': 'tech4',
            'name': 'Lisa Hardware Technician',
            'skills': ['Hardware', 'Network'],
            'experience_months': 48,
            'current_tasks': 3,
            'max_tasks': 5,
            'avg_resolution_time': 4.8,
            'sla_compliance_rate': 0.98,
            'task_completion_rate': 0.99,
            'is_available': True,
            'specialties': {
                'Network': 0.85,
                'Hardware': 0.97,
                'Software': 0.60,
                'Security': 0.75,
                'Email': 0.55
            }
        }
    ]
    
    # Create sample tasks
    tasks = [
        {
            'id': 'task1',
            'title': 'Firewall configuration issue',
            'description': 'Firewall blocking legitimate traffic. Need to update rules.',
            'category': 'Security',
            'priority': 'High',
            'complexity': 'Complex',
            'department': 'IT'
        },
        {
            'id': 'task2',
            'title': 'Outlook email not syncing',
            'description': 'Email not syncing with server. Need to reconfigure Outlook.',
            'category': 'Email',
            'priority': 'Medium',
            'complexity': 'Medium',
            'department': 'HR'
        },
        {
            'id': 'task3',
            'title': 'Printer network connection',
            'description': 'Network printer not connecting to WiFi. Needs troubleshooting.',
            'category': 'Network',
            'priority': 'Low',
            'complexity': 'Simple',
            'department': 'Finance'
        },
        {
            'id': 'task4',
            'title': 'Server hardware failure',
            'description': 'Server showing hardware errors. May need component replacement.',
            'category': 'Hardware',
            'priority': 'Critical',
            'complexity': 'Complex',
            'department': 'IT'
        },
        {
            'id': 'task5',
            'title': 'Software license activation',
            'description': 'Adobe Creative Cloud licenses not activating properly.',
            'category': 'Software',
            'priority': 'Medium',
            'complexity': 'Medium',
            'department': 'Marketing'
        }
    ]
    
    # Create and train matcher
    matcher = TechnicianMatcher()
    
    if not matcher.load_model():
        # Create dummy historical assignments
        historical_assignments = [
            {'technician_id': 'tech1', 'task_category': 'Network', 'success': True},
            {'technician_id': 'tech2', 'task_category': 'Software', 'success': True},
            {'technician_id': 'tech3', 'task_category': 'Security', 'success': True},
            {'technician_id': 'tech4', 'task_category': 'Hardware', 'success': True},
        ]
        
        matcher.train(historical_assignments, technicians)
    
    # Test individual task matching
    print("\nIndividual Task Matching:")
    print("=" * 80)
    
    for task in tasks[:3]:
        print(f"\nTask: {task['title']}")
        print(f"Category: {task['category']}, Priority: {task['priority']}")
        
        matches = matcher.find_best_match(task, top_n=2)
        
        for i, match in enumerate(matches, 1):
            print(f"\n  Match #{i}: {match['technician_name']}")
            print(f"    Total Score: {match['total_score']:.3f}")
            print(f"    Skill Match: {match['scores']['skill_match']:.3f}")
            print(f"    Workload Score: {match['scores']['workload_score']:.3f}")
            print(f"    Experience Score: {match['scores']['experience_score']:.3f}")
            print(f"    Specialty Score: {match['scores']['specialty_score']:.3f}")
    
    # Test batch assignment
    print("\n\nBatch Assignment Recommendation:")
    print("=" * 80)
    
    recommendations = matcher.recommend_assignments(tasks, technicians, strategy='balanced')
    
    print(f"\nAssignment Strategy: {recommendations['statistics']['strategy_used']}")
    print(f"Total Tasks: {recommendations['statistics']['total_tasks']}")
    print(f"Assigned Tasks: {recommendations['statistics']['assigned_tasks']}")
    print(f"Assignment Rate: {recommendations['statistics']['assignment_rate']:.1%}")
    
    print("\nDetailed Assignments:")
    for assignment in recommendations['assignments']:
        if assignment['assigned_to']:
            print(f"\n  Task: {assignment['task_title']}")
            print(f"    Assigned to: {assignment['technician_name']}")
            print(f"    Match Score: {assignment['match_score']:.3f}")
            print(f"    Reason: {assignment['reason']}")
        else:
            print(f"\n  Task: {assignment['task_title']}")
            print(f"    Status: UNASSIGNED")
            print(f"    Reason: {assignment['reason']}")