import joblib

# Load encoder dari file joblib (di-load sekali saja)
le_app_group = joblib.load('label_encoder_app_group.joblib')
le_mother_edu = joblib.load('label_encoder_mother_edu.joblib')
le_father_edu = joblib.load('label_encoder_father_edu.joblib')
le_target = joblib.load('label_encoder_target.joblib')

def group_parent_education(qualification):
    high = {2, 3, 4, 5, 40, 41, 42, 43, 44}
    medium = {1, 9, 10, 12, 13, 14, 18, 19, 20, 22, 25, 27, 31, 33, 39}
    low = {11, 26, 29, 30, 37, 38}
    very_low = {34, 35, 36}
    incomplete = {6}

    if qualification in high:
        return 'high'
    elif qualification in medium:
        return 'medium'
    elif qualification in low:
        return 'low'
    elif qualification in very_low:
        return 'very_low'
    elif qualification in incomplete:
        return 'incomplete'
    else:
        return 'unknown'

def map_application_group(app_mode):
    if app_mode in [1, 17, 18]:
        return 'general_admission'
    elif app_mode in [5, 16]:
        return 'special_regional'
    elif app_mode in [15, 57]:
        return 'international'
    elif app_mode in [7, 44, 53]:
        return 'prior_higher_ed'
    elif app_mode in [42, 43, 51]:
        return 'transfer_or_change'
    elif app_mode in [2, 10, 26, 27]:
        return 'ordinance_entry'
    elif app_mode == 39:
        return 'mature_student'
    else:
        return 'other'

def preprocess_for_model(df, is_train=False):
    df = df.copy()

    # 1. Group parent education
    df['Mother_edu_group'] = df["Mother's qualification"].apply(group_parent_education)
    df['Father_edu_group'] = df["Father's qualification"].apply(group_parent_education)
    df.drop(columns=["Mother's qualification", "Father's qualification"], inplace=True)

    # 2. Map application mode ke group
    df['application_group'] = df['Application mode'].apply(map_application_group)
    df.drop(columns=["Application mode"], inplace=True)

    # 3. Buat fitur rasio & agregat baru
    df['eval_ratio_1st'] = df['Curricular units 1st sem (evaluations)'] / df['Curricular units 1st sem (enrolled)']
    df['approval_rate_1st'] = df['Curricular units 1st sem (approved)'] / df['Curricular units 1st sem (enrolled)']
    df['approval_rate_2nd'] = df['Curricular units 2nd sem (approved)'] / df['Curricular units 2nd sem (enrolled)']
    df['eval_miss_rate_1st'] = df['Curricular units 1st sem (without evaluations)'] / df['Curricular units 1st sem (enrolled)']
    df['total_approved'] = df['Curricular units 1st sem (approved)'] + df['Curricular units 2nd sem (approved)']
    df['total_enrolled'] = df['Curricular units 1st sem (enrolled)'] + df['Curricular units 2nd sem (enrolled)']
    df['overall_approval_rate'] = df['total_approved'] / df['total_enrolled']

    # 4. Drop kolom mentah yang sudah jadi fitur baru
    cols_to_drop = [
        "Curricular units 1st sem (enrolled)",
        "Curricular units 1st sem (evaluations)",
        "Curricular units 1st sem (approved)",
        # "Curricular units 1st sem (grade)",
        "Curricular units 1st sem (without evaluations)",
        "Curricular units 2nd sem (enrolled)",
        "Curricular units 2nd sem (approved)"
    ]
    df.drop(columns=cols_to_drop, inplace=True)

    # 5. Transform fitur kategorikal
    df['application_group'] = le_app_group.transform(df['application_group'])
    df['Mother_edu_group'] = le_mother_edu.transform(df['Mother_edu_group'])
    df['Father_edu_group'] = le_father_edu.transform(df['Father_edu_group'])

    # Hanya encode 'Target' saat training
    if is_train and 'Target' in df.columns:
        df['Target'] = le_target.transform(df['Target'])

    # 6. Urutkan fitur
    feature_order = [
        "Admission grade", "Age at enrollment", "eval_ratio_1st", "approval_rate_1st",  
        "approval_rate_2nd", "eval_miss_rate_1st", "total_approved", "total_enrolled",
        "overall_approval_rate", "Marital status", "application_group", "Daytime/evening attendance\t",
        "Gender", "International", "Scholarship holder", "Tuition fees up to date",	
        "Debtor", "Mother_edu_group", "Father_edu_group"
    ]

    # Tambahkan Target jika ada
    if is_train and 'Target' in df.columns:
        feature_order.append('Target')

    df = df[feature_order]

    return df
