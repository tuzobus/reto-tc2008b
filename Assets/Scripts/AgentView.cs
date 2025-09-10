using UnityEngine;

public class AgentView : MonoBehaviour
{
    [Header("References")]
    public Transform visualRoot;       // hijo "Visual" (donde está el Animator)
    public Animator animator;          // se autollenará si está en visualRoot

    [Header("Tuning")]
    public float yawOffsetDegrees = 0f; // si tu modelo “mira” a +Z deja 0; si mira a +X pon -90
    public float turnSpeed = 12f;       // qué tan rápido rota hacia el movimiento
    public float idleThreshold = 0.1f;  // umbral para considerar Speed=0

    void Awake()
    {
        if (visualRoot == null && transform.childCount > 0)
            visualRoot = transform.GetChild(0);

        if (animator == null && visualRoot != null)
            animator = visualRoot.GetComponentInChildren<Animator>();
    }

    /// Llamado constantemente durante un movimiento para orientar al modelo.
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

    /// Actualiza el parámetro "Speed" del Animator (0 = idle, >0 = caminar)
    public void SetSpeed(float speed)
    {
        if (animator == null) return;
        animator.SetFloat("Speed", (speed < idleThreshold) ? 0f : speed);
    }
}
