using UnityEngine;

public class AgentView : MonoBehaviour
{
    [Header("References")]
    public Transform visualRoot;
    public Animator animator;

    [Header("Tuning")]
    public float yawOffsetDegrees = 0f;
    public float turnSpeed = 12f;
    public float idleThreshold = 0.1f;

    void Awake()
    {
        if (visualRoot == null && transform.childCount > 0)
            visualRoot = transform.GetChild(0);

        if (animator == null && visualRoot != null)
            animator = visualRoot.GetComponentInChildren<Animator>();
    }

    public void FaceDirection(Vector3 worldDelta)
    {
        if (visualRoot == null) return;

        // proyecta a XZ
        worldDelta.y = 0f;
        if (worldDelta.sqrMagnitude < 0.0001f) return;

        Quaternion look = Quaternion.LookRotation(worldDelta.normalized, Vector3.up);
        look = look * Quaternion.Euler(0f, yawOffsetDegrees, 0f);

        visualRoot.rotation = Quaternion.Slerp(visualRoot.rotation, look, Time.deltaTime * turnSpeed);
    }

    public void SetSpeed(float speed)
    {
        if (animator == null) return;
        animator.SetFloat("Speed", (speed < idleThreshold) ? 0f : speed);
    }
}
